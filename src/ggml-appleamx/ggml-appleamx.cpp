// Apple AMX backend
//
// Offloads large MUL_MAT ops to the undocumented Apple AMX matrix coprocessor
// found in M1-M3 Apple silicon. Each CPU cluster has one AMX unit that is
// shared by the cores of that cluster, so multiple threads are used to keep
// all of the units busy.
//
// Both operands are packed into k-major panels on every call (see amx-kernels.h):
//  - F32 / BF16 src0: f32 panels, fma32 into four 16x16 f32 accumulators
//  - F16 / quantized src0: f16 panels, fma16 into one 32x32 f32 accumulator
//
// Weights that are pre-packed when they are loaded are handled by the CPU_APPLEAMX
// buffer type of the CPU backend instead (src/ggml-cpu/apple-amx).

#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-appleamx.h"
#include "ggml-backend-impl.h"

#include "amx-kernels.h"

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

#define AMX_N_BLOCK   16        // src1 panels per work item
#define AMX_K_BLOCK   (16*1024) // max bytes per k block of a panel row

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
