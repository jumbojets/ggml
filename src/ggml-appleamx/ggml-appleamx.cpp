// Apple AMX backend
//
// Offloads large MUL_MAT ops to the undocumented Apple AMX matrix coprocessor
// found in M1-M3 Apple silicon. Each CPU cluster has one AMX unit that is
// shared by the cores of that cluster, so multiple threads are used to keep
// all of the units busy.
//
// The output is computed in 32x32 tiles as a sum of outer products:
//   dst[i1][i0] = sum_k src0[i0][k] * src1[i1][k]
// Both operands are packed into k-major panels of 32 rows, so that for each k
// the 32 values of a panel are contiguous and can be loaded into an AMX X or Y
// register. src0 panels go into X (Z columns, contiguous in dst) and src1
// panels go into Y (Z rows).
//
//  - F32 / BF16 src0: f32 panels, fma32 into four 16x16 f32 accumulators
//  - F16 / quantized src0: f16 panels, fma16 into one 32x32 f32 accumulator

#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-appleamx.h"
#include "ggml-backend-impl.h"

#include "amx.h"

#include <arm_neon.h>
#include <sys/sysctl.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#define AMX_TILE      32        // rows of a packed panel, size of an output tile
#define AMX_N_BLOCK   16        // src1 panels per work item
#define AMX_K_BLOCK   (16*1024) // max bytes per k block of a panel row
#define AMX_ALIGNMENT 128       // pair loads need 128-byte alignment

//
// utils
//

struct amx_aligned_buffer {
    void * data = nullptr;
    size_t size = 0;

    ~amx_aligned_buffer() {
        free(data);
    }

    void * get(size_t desired) {
        if (desired > size) {
            free(data);
            size = GGML_PAD(desired, AMX_ALIGNMENT);
            data = aligned_alloc(AMX_ALIGNMENT, size);
            GGML_ASSERT(data != nullptr);
        }
        return data;
    }
};

// simple persistent thread pool, the calling thread is worker 0
struct amx_thread_pool {
    std::vector<std::thread> workers;
    std::mutex               mutex;
    std::condition_variable  cv_start;
    std::condition_variable  cv_done;

    const std::function<void(int)> * job = nullptr;
    uint64_t generation = 0;
    int      n_pending  = 0;
    bool     stop       = false;

    explicit amx_thread_pool(int n_threads) {
        for (int i = 1; i < n_threads; i++) {
            workers.emplace_back([this, i]() { worker(i); });
        }
    }

    ~amx_thread_pool() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stop = true;
        }
        cv_start.notify_all();
        for (auto & w : workers) {
            w.join();
        }
    }

    int n_threads() const {
        return (int) workers.size() + 1;
    }

    void worker(int ith) {
        uint64_t seen = 0;
        while (true) {
            const std::function<void(int)> * fn;
            {
                std::unique_lock<std::mutex> lock(mutex);
                cv_start.wait(lock, [&]() { return stop || generation != seen; });
                if (stop) {
                    return;
                }
                seen = generation;
                fn   = job;
            }
            (*fn)(ith);
            {
                std::lock_guard<std::mutex> lock(mutex);
                if (--n_pending == 0) {
                    cv_done.notify_one();
                }
            }
        }
    }

    // runs fn(ith) on every thread and waits for all of them to finish
    void run(const std::function<void(int)> & fn) {
        if (workers.empty()) {
            fn(0);
            return;
        }
        {
            std::lock_guard<std::mutex> lock(mutex);
            job       = &fn;
            n_pending = (int) workers.size();
            generation++;
        }
        cv_start.notify_all();
        fn(0);
        std::unique_lock<std::mutex> lock(mutex);
        cv_done.wait(lock, [&]() { return n_pending == 0; });
    }
};

static int amx_sysctl_int(const char * name, int def) {
    int    value = 0;
    size_t size  = sizeof(value);
    if (sysctlbyname(name, &value, &size, nullptr, 0) != 0 || value <= 0) {
        return def;
    }
    return value;
}

// minimum size of every dimension of a MUL_MAT for it to be offloaded
// TODO: find the optimal value
static int64_t amx_min_batch() {
    static const int64_t value = []() {
        const char * env = getenv("GGML_APPLEAMX_MIN_BATCH");
        return env ? std::max<int64_t>(atoll(env), 1) : 32;
    }();
    return value;
}

static int amx_default_n_threads() {
    // one AMX unit per cluster: the performance cores alone already saturate their units
    return amx_sysctl_int("hw.perflevel0.physicalcpu", amx_sysctl_int("hw.physicalcpu", GGML_DEFAULT_N_THREADS));
}

//
// packing
//
// a panel holds up to AMX_TILE rows of length K, stored k-major as [kp][AMX_TILE]
// rows >= nr and columns k >= K are zero
//

// transpose a 4x4 block of f32
static inline void amx_transpose_4x4(float32x4_t & r0, float32x4_t & r1, float32x4_t & r2, float32x4_t & r3) {
    const float32x4_t t0 = vtrn1q_f32(r0, r1);
    const float32x4_t t1 = vtrn2q_f32(r0, r1);
    const float32x4_t t2 = vtrn1q_f32(r2, r3);
    const float32x4_t t3 = vtrn2q_f32(r2, r3);
    r0 = vreinterpretq_f32_f64(vtrn1q_f64(vreinterpretq_f64_f32(t0), vreinterpretq_f64_f32(t2)));
    r1 = vreinterpretq_f32_f64(vtrn1q_f64(vreinterpretq_f64_f32(t1), vreinterpretq_f64_f32(t3)));
    r2 = vreinterpretq_f32_f64(vtrn2q_f64(vreinterpretq_f64_f32(t0), vreinterpretq_f64_f32(t2)));
    r3 = vreinterpretq_f32_f64(vtrn2q_f64(vreinterpretq_f64_f32(t1), vreinterpretq_f64_f32(t3)));
}

// transpose an 8x8 block of 16-bit values
static inline void amx_transpose_8x8(uint16x8_t r[8]) {
    uint16x8_t a[8];
    uint32x4_t b[8];
    for (int i = 0; i < 8; i += 2) {
        a[i + 0] = vtrn1q_u16(r[i], r[i + 1]);
        a[i + 1] = vtrn2q_u16(r[i], r[i + 1]);
    }
    for (int i = 0; i < 8; i += 4) {
        b[i + 0] = vtrn1q_u32(vreinterpretq_u32_u16(a[i + 0]), vreinterpretq_u32_u16(a[i + 2]));
        b[i + 1] = vtrn1q_u32(vreinterpretq_u32_u16(a[i + 1]), vreinterpretq_u32_u16(a[i + 3]));
        b[i + 2] = vtrn2q_u32(vreinterpretq_u32_u16(a[i + 0]), vreinterpretq_u32_u16(a[i + 2]));
        b[i + 3] = vtrn2q_u32(vreinterpretq_u32_u16(a[i + 1]), vreinterpretq_u32_u16(a[i + 3]));
    }
    for (int i = 0; i < 4; i++) {
        r[i + 0] = vreinterpretq_u16_u64(vtrn1q_u64(vreinterpretq_u64_u32(b[i]), vreinterpretq_u64_u32(b[i + 4])));
        r[i + 4] = vreinterpretq_u16_u64(vtrn2q_u64(vreinterpretq_u64_u32(b[i]), vreinterpretq_u64_u32(b[i + 4])));
    }
}

// zero the padding of a panel: rows [nr, AMX_TILE) and columns [K, kp)
template <typename T>
static void amx_pack_pad(T * dst, int nr, int64_t K, int64_t kp) {
    if (nr < AMX_TILE) {
        for (int64_t k = 0; k < K; k++) {
            memset(dst + k*AMX_TILE + nr, 0, (AMX_TILE - nr)*sizeof(T));
        }
    }
    memset(dst + K*AMX_TILE, 0, (kp - K)*AMX_TILE*sizeof(T));
}

// f32 rows -> f32 panel
static void amx_pack_f32_f32(float * dst, const char * src, size_t nb, int nr, int64_t K, int64_t kp) {
    int r = 0;
    for (; r + 4 <= nr; r += 4) {
        const float * s0 = (const float *) (src + (r + 0)*nb);
        const float * s1 = (const float *) (src + (r + 1)*nb);
        const float * s2 = (const float *) (src + (r + 2)*nb);
        const float * s3 = (const float *) (src + (r + 3)*nb);
        int64_t k = 0;
        for (; k + 4 <= K; k += 4) {
            float32x4_t v0 = vld1q_f32(s0 + k);
            float32x4_t v1 = vld1q_f32(s1 + k);
            float32x4_t v2 = vld1q_f32(s2 + k);
            float32x4_t v3 = vld1q_f32(s3 + k);
            amx_transpose_4x4(v0, v1, v2, v3);
            vst1q_f32(dst + (k + 0)*AMX_TILE + r, v0);
            vst1q_f32(dst + (k + 1)*AMX_TILE + r, v1);
            vst1q_f32(dst + (k + 2)*AMX_TILE + r, v2);
            vst1q_f32(dst + (k + 3)*AMX_TILE + r, v3);
        }
        for (; k < K; k++) {
            dst[k*AMX_TILE + r + 0] = s0[k];
            dst[k*AMX_TILE + r + 1] = s1[k];
            dst[k*AMX_TILE + r + 2] = s2[k];
            dst[k*AMX_TILE + r + 3] = s3[k];
        }
    }
    for (; r < nr; r++) {
        const float * s = (const float *) (src + r*nb);
        for (int64_t k = 0; k < K; k++) {
            dst[k*AMX_TILE + r] = s[k];
        }
    }
    amx_pack_pad(dst, nr, K, kp);
}

// f32 rows -> f16 panel
static void amx_pack_f32_f16(ggml_fp16_t * dst_, const char * src, size_t nb, int nr, int64_t K, int64_t kp) {
    __fp16 * dst = (__fp16 *) dst_;
    int r = 0;
    for (; r + 4 <= nr; r += 4) {
        const float * s0 = (const float *) (src + (r + 0)*nb);
        const float * s1 = (const float *) (src + (r + 1)*nb);
        const float * s2 = (const float *) (src + (r + 2)*nb);
        const float * s3 = (const float *) (src + (r + 3)*nb);
        int64_t k = 0;
        for (; k + 4 <= K; k += 4) {
            float32x4_t v0 = vld1q_f32(s0 + k);
            float32x4_t v1 = vld1q_f32(s1 + k);
            float32x4_t v2 = vld1q_f32(s2 + k);
            float32x4_t v3 = vld1q_f32(s3 + k);
            amx_transpose_4x4(v0, v1, v2, v3);
            vst1_f16(dst + (k + 0)*AMX_TILE + r, vcvt_f16_f32(v0));
            vst1_f16(dst + (k + 1)*AMX_TILE + r, vcvt_f16_f32(v1));
            vst1_f16(dst + (k + 2)*AMX_TILE + r, vcvt_f16_f32(v2));
            vst1_f16(dst + (k + 3)*AMX_TILE + r, vcvt_f16_f32(v3));
        }
        for (; k < K; k++) {
            dst[k*AMX_TILE + r + 0] = (__fp16) s0[k];
            dst[k*AMX_TILE + r + 1] = (__fp16) s1[k];
            dst[k*AMX_TILE + r + 2] = (__fp16) s2[k];
            dst[k*AMX_TILE + r + 3] = (__fp16) s3[k];
        }
    }
    for (; r < nr; r++) {
        const float * s = (const float *) (src + r*nb);
        for (int64_t k = 0; k < K; k++) {
            dst[k*AMX_TILE + r] = (__fp16) s[k];
        }
    }
    amx_pack_pad(dst, nr, K, kp);
}

// f16 rows -> f16 panel
static void amx_pack_f16_f16(ggml_fp16_t * dst, const char * src, size_t nb, int nr, int64_t K, int64_t kp) {
    int r = 0;
    for (; r + 8 <= nr; r += 8) {
        const uint16_t * s[8];
        for (int i = 0; i < 8; i++) {
            s[i] = (const uint16_t *) (src + (r + i)*nb);
        }
        int64_t k = 0;
        for (; k + 8 <= K; k += 8) {
            uint16x8_t v[8];
            for (int i = 0; i < 8; i++) {
                v[i] = vld1q_u16(s[i] + k);
            }
            amx_transpose_8x8(v);
            for (int i = 0; i < 8; i++) {
                vst1q_u16(dst + (k + i)*AMX_TILE + r, v[i]);
            }
        }
        for (; k < K; k++) {
            for (int i = 0; i < 8; i++) {
                dst[k*AMX_TILE + r + i] = s[i][k];
            }
        }
    }
    for (; r < nr; r++) {
        const ggml_fp16_t * s = (const ggml_fp16_t *) (src + r*nb);
        for (int64_t k = 0; k < K; k++) {
            dst[k*AMX_TILE + r] = s[k];
        }
    }
    amx_pack_pad(dst, nr, K, kp);
}

//
// micro kernels
//
// compute a 32x32 tile of dst from an src0 panel (X) and an src1 panel (Y)
// tile row j (src1 row) is stored at c + j*ldc, only the first nr rows and nc columns are stored
// with acc, the result is added to the existing contents of the tile
//

static void amx_kernel_f32(const float * a, const float * b, float * c, size_t ldc, int64_t kp, int nr, int nc, bool acc) {
    // z row 4*j + q holds columns [16*(q%2), 16*(q%2) + 16) of tile row j + 16*(q/2)
    if (acc) {
        if (nr == AMX_TILE && nc == AMX_TILE) {
            for (int j = 0; j < 16; j++) {
                AMX_LDZ(c + (j +  0)*ldc +  0, 4*j + 0, 0);
                AMX_LDZ(c + (j +  0)*ldc + 16, 4*j + 1, 0);
                AMX_LDZ(c + (j + 16)*ldc +  0, 4*j + 2, 0);
                AMX_LDZ(c + (j + 16)*ldc + 16, 4*j + 3, 0);
            }
        } else {
            alignas(AMX_ALIGNMENT) float row[AMX_TILE] = {0};
            for (int j = 0; j < nr; j++) {
                const int q = j < 16 ? 0 : 2;
                memcpy(row, c + j*ldc, nc*sizeof(float));
                AMX_LDZ(row +  0, 4*(j % 16) + q + 0, 0);
                AMX_LDZ(row + 16, 4*(j % 16) + q + 1, 0);
            }
        }
    }

    // kp is a multiple of 4
    for (int64_t k = 0; k < kp; k += 4) {
        const float * ak = a + k*AMX_TILE;
        const float * bk = b + k*AMX_TILE;
        AMX_LDX(ak + 0*AMX_TILE, 0, AMX_LDST_PAIR);
        AMX_LDY(bk + 0*AMX_TILE, 0, AMX_LDST_PAIR);
        AMX_LDX(ak + 1*AMX_TILE, 2, AMX_LDST_PAIR);
        AMX_LDY(bk + 1*AMX_TILE, 2, AMX_LDST_PAIR);
        AMX_LDX(ak + 2*AMX_TILE, 4, AMX_LDST_PAIR);
        AMX_LDY(bk + 2*AMX_TILE, 4, AMX_LDST_PAIR);
        AMX_LDX(ak + 3*AMX_TILE, 6, AMX_LDST_PAIR);
        AMX_LDY(bk + 3*AMX_TILE, 6, AMX_LDST_PAIR);

        // without acc, the first iteration overwrites Z instead of accumulating
        uint64_t flags = k == 0 && !acc ? AMX_SKIP_Z : 0;
        for (int u = 0; u < 4; u++) {
            AMX_FMA32(u*128 +  0, u*128 +  0, 0, flags);
            AMX_FMA32(u*128 +  0, u*128 + 64, 1, flags);
            AMX_FMA32(u*128 + 64, u*128 +  0, 2, flags);
            AMX_FMA32(u*128 + 64, u*128 + 64, 3, flags);
            flags = 0;
        }
    }

    if (nr == AMX_TILE && nc == AMX_TILE) {
        for (int j = 0; j < 16; j++) {
            AMX_STZ(c + (j +  0)*ldc +  0, 4*j + 0, 0);
            AMX_STZ(c + (j +  0)*ldc + 16, 4*j + 1, 0);
            AMX_STZ(c + (j + 16)*ldc +  0, 4*j + 2, 0);
            AMX_STZ(c + (j + 16)*ldc + 16, 4*j + 3, 0);
        }
    } else {
        alignas(AMX_ALIGNMENT) float row[AMX_TILE];
        for (int j = 0; j < nr; j++) {
            const int q = j < 16 ? 0 : 2;
            AMX_STZ(row +  0, 4*(j % 16) + q + 0, 0);
            AMX_STZ(row + 16, 4*(j % 16) + q + 1, 0);
            memcpy(c + j*ldc, row, nc*sizeof(float));
        }
    }
}

static void amx_kernel_f16(const ggml_fp16_t * a, const ggml_fp16_t * b, float * c, size_t ldc, int64_t kp, int nr, int nc, bool acc) {
    // ldzi/stzi (2*j + h) load/store columns [16*h, 16*h + 16) of tile row j
    if (acc) {
        if (nr == AMX_TILE && nc == AMX_TILE) {
            for (int j = 0; j < AMX_TILE; j++) {
                AMX_LDZI(c + j*ldc +  0, 2*j + 0, 0);
                AMX_LDZI(c + j*ldc + 16, 2*j + 1, 0);
            }
        } else {
            alignas(AMX_ALIGNMENT) float row[AMX_TILE] = {0};
            for (int j = 0; j < nr; j++) {
                memcpy(row, c + j*ldc, nc*sizeof(float));
                AMX_LDZI(row +  0, 2*j + 0, 0);
                AMX_LDZI(row + 16, 2*j + 1, 0);
            }
        }
    }

    // kp is a multiple of 8
    for (int64_t k = 0; k < kp; k += 8) {
        const ggml_fp16_t * ak = a + k*AMX_TILE;
        const ggml_fp16_t * bk = b + k*AMX_TILE;
        AMX_LDX(ak + 0*AMX_TILE, 0, AMX_LDST_PAIR);
        AMX_LDY(bk + 0*AMX_TILE, 0, AMX_LDST_PAIR);
        AMX_LDX(ak + 2*AMX_TILE, 2, AMX_LDST_PAIR);
        AMX_LDY(bk + 2*AMX_TILE, 2, AMX_LDST_PAIR);
        AMX_LDX(ak + 4*AMX_TILE, 4, AMX_LDST_PAIR);
        AMX_LDY(bk + 4*AMX_TILE, 4, AMX_LDST_PAIR);
        AMX_LDX(ak + 6*AMX_TILE, 6, AMX_LDST_PAIR);
        AMX_LDY(bk + 6*AMX_TILE, 6, AMX_LDST_PAIR);

        uint64_t flags = k == 0 && !acc ? AMX_SKIP_Z : 0;
        for (int u = 0; u < 8; u++) {
            AMX_FMA16(u*64, u*64, 0, flags | AMX_Z_F32);
            flags = 0;
        }
    }

    if (nr == AMX_TILE && nc == AMX_TILE) {
        for (int j = 0; j < AMX_TILE; j++) {
            AMX_STZI(c + j*ldc +  0, 2*j + 0, 0);
            AMX_STZI(c + j*ldc + 16, 2*j + 1, 0);
        }
    } else {
        alignas(AMX_ALIGNMENT) float row[AMX_TILE];
        for (int j = 0; j < nr; j++) {
            AMX_STZI(row +  0, 2*j + 0, 0);
            AMX_STZI(row + 16, 2*j + 1, 0);
            memcpy(c + j*ldc, row, nc*sizeof(float));
        }
    }
}

//
// backend context
//

struct ggml_backend_appleamx_context {
    int n_threads = amx_default_n_threads();

    std::unique_ptr<amx_thread_pool> pool;

    amx_aligned_buffer              wdata;   // packed src1
    std::vector<amx_aligned_buffer> wpanel;  // per-thread packed src0 panel
    std::vector<amx_aligned_buffer> wrows;   // per-thread src0 rows converted to f32

    amx_thread_pool & get_pool() {
        if (!pool || pool->n_threads() != n_threads) {
            pool.reset();
            pool.reset(new amx_thread_pool(n_threads));
            wpanel.resize(n_threads);
            wrows.resize(n_threads);
        }
        return *pool;
    }
};

static bool amx_use_f16(enum ggml_type type) {
    // f32 and bf16 keep f32 precision, everything else is computed with f16 operands
    return type != GGML_TYPE_F32 && type != GGML_TYPE_BF16;
}

static void ggml_backend_appleamx_mul_mat(ggml_backend_appleamx_context * ctx, struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    const enum ggml_type type = src0->type;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    GGML_ASSERT(nb00 == ggml_type_size(type));
    GGML_ASSERT(nb10 == sizeof(float));
    GGML_ASSERT(nb0  == sizeof(float));
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    // broadcast factors
    const int64_t r2 = ne12/ne02;
    const int64_t r3 = ne13/ne03;

    const bool   use_f16 = amx_use_f16(type);
    const size_t esize   = use_f16 ? sizeof(ggml_fp16_t) : sizeof(float);

    const int64_t K  = ne00;
    const int64_t kp = use_f16 ? GGML_PAD(K, 8) : GGML_PAD(K, 4);

    // long rows are split in K blocks to keep the working set of a work item in the L2 cache
    // reloading the accumulators for every block is not free, so the blocks are large
    // the block size must be a multiple of the quantization block size and of 8
    const int64_t kb_align   = std::lcm<int64_t>(8, ggml_blck_size(type));
    const int64_t kb_max     = std::max<int64_t>(AMX_K_BLOCK/esize/kb_align*kb_align, kb_align);
    const int64_t kb_count   = (K + kb_max - 1)/kb_max;
    const int64_t kb_size    = ((K + kb_count - 1)/kb_count + kb_align - 1)/kb_align*kb_align;
    const int64_t n_kb       = (K + kb_size - 1)/kb_size;

    const size_t  panel_size = kp*AMX_TILE*esize;
    const int64_t n_mp       = (ne01 + AMX_TILE - 1)/AMX_TILE;  // src0 panels per plane
    const int64_t n_np       = (ne11 + AMX_TILE - 1)/AMX_TILE;  // src1 panels per plane
    const int64_t n_nb       = (n_np + AMX_N_BLOCK - 1)/AMX_N_BLOCK;
    const int64_t n_planes   = ne12*ne13;

    amx_thread_pool & pool = ctx->get_pool();

    char * wdata = (char *) ctx->wdata.get(n_planes*n_np*panel_size);

    const ggml_to_float_t to_float = ggml_get_type_traits(type)->to_float;
    const bool needs_to_float = type != GGML_TYPE_F32 && type != GGML_TYPE_F16;

    // pack src1
    {
        std::atomic<int64_t> next(0);
        const int64_t n_items = n_planes*n_np;

        pool.run([&](int) {
            for (int64_t item = next++; item < n_items; item = next++) {
                const int64_t ip  = item/n_np;
                const int64_t inp = item%n_np;
                const int64_t i12 = ip%ne12;
                const int64_t i13 = ip/ne12;
                const int64_t i11 = inp*AMX_TILE;
                const int     nr  = (int) std::min<int64_t>(AMX_TILE, ne11 - i11);

                const char * src = (const char *) src1->data + i11*nb11 + i12*nb12 + i13*nb13;
                void       * pp  = wdata + item*panel_size;

                if (use_f16) {
                    amx_pack_f32_f16((ggml_fp16_t *) pp, src, nb11, nr, K, kp);
                } else {
                    amx_pack_f32_f32((float *) pp, src, nb11, nr, K, kp);
                }
            }
        });
    }

    // compute
    {
        std::atomic<int64_t> next(0);
        // ordered so that concurrent work items share the same block of src1 panels
        const int64_t n_items = n_planes*n_nb*n_mp;

        pool.run([&](int ith) {
            char  * panel = (char *) ctx->wpanel[ith].get(std::min(kp, kb_size)*AMX_TILE*esize);
            float * rows  = needs_to_float ? (float *) ctx->wrows[ith].get(AMX_TILE*kb_size*sizeof(float)) : nullptr;

            bool amx_enabled = false;

            for (int64_t item = next++; item < n_items; item = next++) {
                const int64_t imp = item%n_mp;
                const int64_t inb = (item/n_mp)%n_nb;
                const int64_t ip  = item/(n_mp*n_nb);
                const int64_t i12 = ip%ne12;
                const int64_t i13 = ip/ne12;
                const int64_t i02 = i12/r2;
                const int64_t i03 = i13/r3;
                const int64_t i01 = imp*AMX_TILE;
                const int     nc  = (int) std::min<int64_t>(AMX_TILE, ne01 - i01);

                if (!amx_enabled) {
                    AMX_SET();
                    amx_enabled = true;
                }

                const char * src = (const char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03;

                for (int64_t ikb = 0; ikb < n_kb; ikb++) {
                    const int64_t k0   = ikb*kb_size;
                    const int64_t kn   = std::min(kb_size, K - k0);  // valid k values in this block
                    const int64_t knp  = std::min(kb_size, kp - k0); // including padding
                    const char  * srck = src + (k0/ggml_blck_size(type))*ggml_type_size(type);

                    // pack this block of the src0 panel
                    if (needs_to_float) {
                        for (int r = 0; r < nc; r++) {
                            to_float(srck + r*nb01, rows + r*kn, kn);
                        }
                        if (use_f16) {
                            amx_pack_f32_f16((ggml_fp16_t *) panel, (const char *) rows, kn*sizeof(float), nc, kn, knp);
                        } else {
                            amx_pack_f32_f32((float *) panel, (const char *) rows, kn*sizeof(float), nc, kn, knp);
                        }
                    } else if (use_f16) {
                        amx_pack_f16_f16((ggml_fp16_t *) panel, srck, nb01, nc, kn, knp);
                    } else {
                        amx_pack_f32_f32((float *) panel, srck, nb01, nc, kn, knp);
                    }

                    const int64_t inp0 = inb*AMX_N_BLOCK;
                    const int64_t inp1 = std::min(n_np, inp0 + AMX_N_BLOCK);
                    for (int64_t inp = inp0; inp < inp1; inp++) {
                        const int64_t i11 = inp*AMX_TILE;
                        const int     nr  = (int) std::min<int64_t>(AMX_TILE, ne11 - i11);

                        const char * pb = wdata + (ip*n_np + inp)*panel_size + k0*AMX_TILE*esize;
                        float      * c  = (float *) ((char *) dst->data + i01*nb0 + i11*nb1 + i12*nb2 + i13*nb3);

                        if (use_f16) {
                            amx_kernel_f16((const ggml_fp16_t *) panel, (const ggml_fp16_t *) pb, c, nb1/sizeof(float), knp, nr, nc, ikb > 0);
                        } else {
                            amx_kernel_f32((const float *) panel, (const float *) pb, c, nb1/sizeof(float), knp, nr, nc, ikb > 0);
                        }
                    }
                }
            }

            if (amx_enabled) {
                AMX_CLR();
            }
        });
    }

}

//
// backend interface
//

static const char * ggml_backend_appleamx_get_name(ggml_backend_t backend) {
    return "AppleAMX";

    GGML_UNUSED(backend);
}

static void ggml_backend_appleamx_free(ggml_backend_t backend) {
    ggml_backend_appleamx_context * ctx = (ggml_backend_appleamx_context *) backend->context;
    delete ctx;
    delete backend;
}

static enum ggml_status ggml_backend_appleamx_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    ggml_backend_appleamx_context * ctx = (ggml_backend_appleamx_context *) backend->context;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }

        switch (node->op) {
            case GGML_OP_MUL_MAT:
                ggml_backend_appleamx_mul_mat(ctx, node);
                break;

            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                break;

            default:
                GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(node));
        }
    }

    return GGML_STATUS_SUCCESS;
}

static struct ggml_backend_i appleamx_backend_i = {
    /* .get_name                = */ ggml_backend_appleamx_get_name,
    /* .free                    = */ ggml_backend_appleamx_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_appleamx_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

static ggml_guid_t ggml_backend_appleamx_guid(void) {
    static ggml_guid guid = { 0x5b, 0x2e, 0x93, 0x0c, 0x7a, 0x41, 0x4f, 0xd8, 0xa6, 0x1f, 0xe2, 0x37, 0x9c, 0x05, 0xb4, 0x6a };
    return &guid;
}

ggml_backend_t ggml_backend_appleamx_init(void) {
    ggml_backend_appleamx_context * ctx = new ggml_backend_appleamx_context;

    ggml_backend_t backend = new ggml_backend {
        /* .guid    = */ ggml_backend_appleamx_guid(),
        /* .iface   = */ appleamx_backend_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_appleamx_reg(), 0),
        /* .context = */ ctx,
    };

    return backend;
}

bool ggml_backend_is_appleamx(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_appleamx_guid());
}

void ggml_backend_appleamx_set_n_threads(ggml_backend_t backend_appleamx, int n_threads) {
    GGML_ASSERT(ggml_backend_is_appleamx(backend_appleamx));

    ggml_backend_appleamx_context * ctx = (ggml_backend_appleamx_context *) backend_appleamx->context;
    ctx->n_threads = std::max(n_threads, 1);
}

//
// device interface
//

static const char * ggml_backend_appleamx_device_get_name(ggml_backend_dev_t dev) {
    return "AppleAMX";

    GGML_UNUSED(dev);
}

static const char * ggml_backend_appleamx_device_get_description(ggml_backend_dev_t dev) {
    static const std::string description = []() {
        char   brand[128] = {0};
        size_t size       = sizeof(brand) - 1;
        if (sysctlbyname("machdep.cpu.brand_string", brand, &size, nullptr, 0) != 0) {
            return std::string("Apple AMX");
        }
        return std::string(brand) + " AMX";
    }();
    return description.c_str();

    GGML_UNUSED(dev);
}

static void ggml_backend_appleamx_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    // no memory to report
    *free  = 0;
    *total = 0;

    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_appleamx_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;

    GGML_UNUSED(dev);
}

static void ggml_backend_appleamx_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_appleamx_device_get_name(dev);
    props->description = ggml_backend_appleamx_device_get_description(dev);
    props->type        = ggml_backend_appleamx_device_get_type(dev);
    ggml_backend_appleamx_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ true,
        /* .events                = */ false,
        /* .mmap_support          = */ true,
    };
}

static ggml_backend_t ggml_backend_appleamx_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    return ggml_backend_appleamx_init();

    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_appleamx_device_get_buffer_type(ggml_backend_dev_t dev) {
    return ggml_backend_cpu_buffer_type();

    GGML_UNUSED(dev);
}

static ggml_backend_buffer_t ggml_backend_appleamx_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    return ggml_backend_cpu_buffer_from_ptr(ptr, size);

    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
}

static bool ggml_backend_appleamx_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;

        case GGML_OP_MUL_MAT:
        {
            const struct ggml_tensor * src0 = op->src[0];
            const struct ggml_tensor * src1 = op->src[1];

            // packing the operands is only worth it for large enough matrices
            const int64_t min_batch = amx_min_batch();

            // default back to CPU fast path
            // see: https://github.com/ggml-org/llama.cpp/issues/25565
            if (ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                return false;
            }

            const bool type_ok = src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 ||
                                 ((ggml_is_quantized(src0->type) || src0->type == GGML_TYPE_BF16) &&
                                  ggml_get_type_traits(src0->type)->to_float != NULL);

            return type_ok &&
                   src1->type == GGML_TYPE_F32 &&
                   op->type   == GGML_TYPE_F32 &&
                   src0->nb[0] == ggml_type_size(src0->type) &&
                   src1->nb[0] == sizeof(float) &&
                   op->nb[0]   == sizeof(float) &&
                   op->ne[0] >= min_batch && op->ne[1] >= min_batch && src1->ne[0] >= min_batch;
        }

        default:
            return false;
    }

    GGML_UNUSED(dev);
}

static bool ggml_backend_appleamx_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_is_host(buft);

    GGML_UNUSED(dev);
}

static const struct ggml_backend_device_i ggml_backend_appleamx_device_i = {
    /* .get_name             = */ ggml_backend_appleamx_device_get_name,
    /* .get_description      = */ ggml_backend_appleamx_device_get_description,
    /* .get_memory           = */ ggml_backend_appleamx_device_get_memory,
    /* .get_type             = */ ggml_backend_appleamx_device_get_type,
    /* .get_props            = */ ggml_backend_appleamx_device_get_props,
    /* .init_backend         = */ ggml_backend_appleamx_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_appleamx_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ ggml_backend_appleamx_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_appleamx_device_supports_op,
    /* .supports_buft        = */ ggml_backend_appleamx_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

//
// backend reg interface
//

static const char * ggml_backend_appleamx_reg_get_name(ggml_backend_reg_t reg) {
    return "AppleAMX";

    GGML_UNUSED(reg);
}

static size_t ggml_backend_appleamx_reg_get_device_count(ggml_backend_reg_t reg) {
    return 1;

    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_appleamx_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    static ggml_backend_device ggml_backend_appleamx_device = {
        /* .iface   = */ ggml_backend_appleamx_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };

    return &ggml_backend_appleamx_device;

    GGML_UNUSED(index);
}

static void * ggml_backend_appleamx_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    if (std::strcmp(name, "ggml_backend_set_n_threads") == 0) {
        return (void *) ggml_backend_appleamx_set_n_threads;
    }
    return NULL;

    GGML_UNUSED(reg);
}

static const struct ggml_backend_reg_i ggml_backend_appleamx_reg_i = {
    /* .get_name         = */ ggml_backend_appleamx_reg_get_name,
    /* .get_device_count = */ ggml_backend_appleamx_reg_get_device_count,
    /* .get_device       = */ ggml_backend_appleamx_reg_get_device,
    /* .get_proc_address = */ ggml_backend_appleamx_get_proc_address,
};

ggml_backend_reg_t ggml_backend_appleamx_reg(void) {
    static struct ggml_backend_reg ggml_backend_appleamx_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_appleamx_reg_i,
        /* .context     = */ NULL,
    };

    return &ggml_backend_appleamx_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_appleamx_reg)
