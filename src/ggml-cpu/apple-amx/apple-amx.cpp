// Apple AMX buffer type of the CPU backend
//
// MUL_MAT weights are packed once, when they are loaded, into panels of 32 rows:
//
//  - F16: k-major f16 panels [kp][32], the layout the AMX kernels consume
//  - quantized types: per block of 32 k values, the quants followed by 32 f16 scales per row and,
//    depending on the type, a second set of scales or 32 f16 mins
//
//    type        quants        decoded weight               bits per weight
//    Q4_0        4-bit         w = s*(q - 8)                4.5
//    Q4_1, Q4_K  4-bit         w = s*q + m                  5
//    Q5_0        4 + 1 bits    w = s*(q - 16)               5.5
//    Q5_1, Q5_K  4 + 1 bits    w = s*q + m                  6
//    Q6_K        4 + 2 bits    w = s*(q - 32), with separate scales for k [0, 16) and [16, 32)   7
//    Q8_0        8-bit         w = s*q                      8.5
//
// The quants of a block are stored in 8 groups of 4 consecutive k values, row by row:
//  - 4-bit: 64 bytes per group, byte 4*r + kk holds row r in the low and row r + 16 in the high nibble,
//    so that the 16-byte vector v of a group holds the rows [4*v, 4*v + 4) x 4 k values
//  - higher bits: 1 or 2 bit planes of 16 bytes per group after the 4-bit quants; bit v of byte b of
//    plane h is bit 4 + h of the element b of vector v
//  - 8-bit: 128 bytes per group, byte 4*r + kk holds row r
// so that the decoded vectors are ready for the NEON dot product instructions,
// and vld4 de-interleaves a group into k-major vectors for the AMX panels.
//
// Small batches are computed with NEON dot products against int8 quantized activations.
// Large batches dequantize the weight panels to f16 and use the AMX fma16 kernel.

#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml-cpu.h"
#include "ggml-cpu-impl.h"
#include "traits.h"

#include "apple-amx.h"
#include "ggml-appleamx/amx-kernels.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

// batch sizes from which the AMX kernels are used instead of the NEON ones
// the AMX kernels have a fixed cost up to a full tile of 32 activation rows, the NEON ones grow with each row
#define AMX_MIN_BATCH_F16 8
#define AMX_MIN_BATCH_Q   16

#define AMX_QK        32        // k values per quantization block
#define AMX_K_BLOCK   (16*1024) // max bytes per k block of an f16 panel row

namespace ggml::cpu::appleamx {

//
// packed layout
//

enum format {
    FORMAT_NONE,
    FORMAT_F16,
    FORMAT_Q,   // quantized, see kind
};

// packed quantized formats
enum kind {
    KIND_Q4,    // Q4_0
    KIND_Q4M,   // Q4_1, Q4_K
    KIND_Q5,    // Q5_0
    KIND_Q5M,   // Q5_1, Q5_K
    KIND_Q6,    // Q6_K
    KIND_Q8,    // Q8_0
};

static constexpr bool kind_q8     (kind k) { return k == KIND_Q8; }
static constexpr int  kind_nhi    (kind k) { return k == KIND_Q5 || k == KIND_Q5M ? 1 : k == KIND_Q6 ? 2 : 0; } // high bit planes
static constexpr bool kind_has_min(kind k) { return k == KIND_Q4M || k == KIND_Q5M; }
static constexpr bool kind_split  (kind k) { return k == KIND_Q6; }                                     // scales per half block
static constexpr int  kind_offset (kind k) { return k == KIND_Q5 ? 16 : k == KIND_Q6 ? 32 : 0; }        // subtracted from q
static constexpr bool kind_min8   (kind k) { return k == KIND_Q4; }                                     // implied min -8*s

#define AMX_QNIB   (AMX_QK*AMX_TILE/2)                 // bytes of 4-bit quants per block
#define AMX_QPLANE (AMX_QK*AMX_TILE/8)                 // bytes per high bit plane per block
#define AMX_QSCALE ((size_t) AMX_TILE*sizeof(ggml_fp16_t)) // bytes per set of scales or mins

struct layout {
    format  fmt        = FORMAT_NONE;
    kind    knd        = KIND_Q4;
    int64_t K          = 0; // row length
    int64_t M          = 0; // number of rows
    int64_t kp         = 0; // padded row length
    size_t  qsize      = 0; // bytes of quants per block
    size_t  block_size = 0; // bytes per block of AMX_QK k values of a panel
    size_t  panel_size = 0; // bytes per panel of AMX_TILE rows
    int64_t n_panels   = 0;

    size_t size() const {
        return n_panels*panel_size;
    }

    // offsets in a block of the scales of each half and of the mins
    size_t scale_offset(int half) const {
        return qsize + (kind_split(knd) ? half : 0)*AMX_QSCALE;
    }
    size_t min_offset() const {
        return qsize + (kind_split(knd) ? 2 : 1)*AMX_QSCALE;
    }
};

static layout get_layout(const struct ggml_tensor * t) {
    layout l;

    if (t->ne[2] != 1 || t->ne[3] != 1) {
        return l;
    }

    l.fmt = FORMAT_Q;
    switch (t->type) {
        case GGML_TYPE_F16:  l.fmt = FORMAT_F16;  break;
        case GGML_TYPE_Q4_0: l.knd = KIND_Q4;     break;
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q4_K: l.knd = KIND_Q4M;    break;
        case GGML_TYPE_Q5_0: l.knd = KIND_Q5;     break;
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q5_K: l.knd = KIND_Q5M;    break;
        case GGML_TYPE_Q6_K: l.knd = KIND_Q6;     break;
        case GGML_TYPE_Q8_0: l.knd = KIND_Q8;     break;
        default:
            l.fmt = FORMAT_NONE;
            return l;
    }

    l.K        = t->ne[0];
    l.M        = t->ne[1];
    l.n_panels = (l.M + AMX_TILE - 1)/AMX_TILE;

    if (l.fmt == FORMAT_F16) {
        l.kp         = GGML_PAD(l.K, 8);
        l.panel_size = l.kp*AMX_TILE*sizeof(ggml_fp16_t);
    } else {
        // the row length of these types is a multiple of their block size, itself a multiple of AMX_QK
        GGML_ASSERT(l.K % AMX_QK == 0);
        l.kp         = l.K;
        l.qsize      = kind_q8(l.knd) ? 2*AMX_QNIB : AMX_QNIB + kind_nhi(l.knd)*AMX_QPLANE;
        l.block_size = l.min_offset() + (kind_has_min(l.knd) ? AMX_QSCALE : 0);
        l.panel_size = (l.K/AMX_QK)*l.block_size;
    }

    return l;
}

static inline float fp16_to_fp32(ggml_fp16_t h) {
    __fp16 f;
    memcpy(&f, &h, sizeof(f));
    return f;
}

static inline ggml_fp16_t fp32_to_fp16(float x) {
    const __fp16 f = x;
    ggml_fp16_t h;
    memcpy(&h, &f, sizeof(h));
    return h;
}

//
// packing
//

static inline void get_scale_min_k4(int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >>  4) | ((q[j - 0] >> 6) << 4);
    }
}

// quants, scales and min of the block ib (of AMX_QK k values) of a row
// the quants of Q5_0 and Q6_K are returned without their offset
// s1 is the scale of the second half of the block, only different from s for Q6_K
static void extract_block(enum ggml_type type, const void * row, int64_t ib, int8_t * q, float & s, float & s1, float & m) {
    m = 0.0f;
    switch (type) {
        case GGML_TYPE_Q4_0:
            {
                const block_q4_0 * b = (const block_q4_0 *) row + ib;
                for (int j = 0; j < 16; j++) {
                    q[j +  0] = b->qs[j] & 0xF;
                    q[j + 16] = b->qs[j] >> 4;
                }
                s = fp16_to_fp32(b->d);
            } break;
        case GGML_TYPE_Q4_1:
            {
                const block_q4_1 * b = (const block_q4_1 *) row + ib;
                for (int j = 0; j < 16; j++) {
                    q[j +  0] = b->qs[j] & 0xF;
                    q[j + 16] = b->qs[j] >> 4;
                }
                s = fp16_to_fp32(b->GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d);
                m = fp16_to_fp32(b->GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.m);
            } break;
        case GGML_TYPE_Q4_K:
            {
                const block_q4_K * b = (const block_q4_K *) row + ib/8;
                const int j = ib%8;
                uint8_t sc, mn;
                get_scale_min_k4(j, b->scales, &sc, &mn);
                const uint8_t * qs = b->qs + 32*(j/2);
                for (int l = 0; l < 32; l++) {
                    q[l] = j%2 == 0 ? qs[l] & 0xF : qs[l] >> 4;
                }
                s =  fp16_to_fp32(b->GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d)*sc;
                m = -fp16_to_fp32(b->GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin)*mn;
            } break;
        case GGML_TYPE_Q8_0:
            {
                const block_q8_0 * b = (const block_q8_0 *) row + ib;
                memcpy(q, b->qs, 32);
                s = fp16_to_fp32(b->d);
            } break;
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
            {
                const uint8_t * qh_p;
                const uint8_t * qs;
                if (type == GGML_TYPE_Q5_0) {
                    const block_q5_0 * b = (const block_q5_0 *) row + ib;
                    qh_p = b->qh;
                    qs   = b->qs;
                    s    = fp16_to_fp32(b->d);
                } else {
                    const block_q5_1 * b = (const block_q5_1 *) row + ib;
                    qh_p = b->qh;
                    qs   = b->qs;
                    s    = fp16_to_fp32(b->GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d);
                    m    = fp16_to_fp32(b->GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.m);
                }
                uint32_t qh;
                memcpy(&qh, qh_p, sizeof(qh));
                for (int j = 0; j < 16; j++) {
                    q[j +  0] = (qs[j] & 0xF) | (((qh >> (j +  0)) << 4) & 0x10);
                    q[j + 16] = (qs[j] >>  4) | (((qh >> (j + 12))     ) & 0x10);
                }
            } break;
        case GGML_TYPE_Q5_K:
            {
                const block_q5_K * b = (const block_q5_K *) row + ib/8;
                const int j = ib%8;
                uint8_t sc, mn;
                get_scale_min_k4(j, b->scales, &sc, &mn);
                const uint8_t * qs = b->qs + 32*(j/2);
                for (int l = 0; l < 32; l++) {
                    q[l] = (j%2 == 0 ? qs[l] & 0xF : qs[l] >> 4) + (((b->qh[l] >> j) & 1) << 4);
                }
                s =  fp16_to_fp32(b->GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d)*sc;
                m = -fp16_to_fp32(b->GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin)*mn;
            } break;
        case GGML_TYPE_Q6_K:
            {
                // a super-block holds 2 halves of 128 values, each of 4 blocks t of 32 values
                const block_q6_K * b = (const block_q6_K *) row + ib/8;
                const int n = (ib%8)/4;
                const int t = ib%4;
                const uint8_t * ql = b->ql + 64*n + (t & 1)*32;
                const uint8_t * qh = b->qh + 32*n;
                const int8_t  * sc = b->scales + 8*n + 2*t;
                for (int l = 0; l < 32; l++) {
                    q[l] = ((ql[l] >> ((t & 2) ? 4 : 0)) & 0xF) | (((qh[l] >> (2*t)) & 3) << 4);
                }
                const float d = fp16_to_fp32(b->d);
                s  = d*sc[0];
                s1 = d*sc[1];
                return;
            }
        default:
            GGML_ABORT("unsupported type");
    }
    s1 = s;
}

static void pack_panel(const struct ggml_tensor * t, const layout & l, const char * src, int64_t ip) {
    const int64_t i01 = ip*AMX_TILE;
    const int     nr  = (int) std::min<int64_t>(AMX_TILE, l.M - i01);

    char * dst = (char *) t->data + ip*l.panel_size;

    if (l.fmt == FORMAT_F16) {
        amx_pack_f16_f16((ggml_fp16_t *) dst, src + i01*t->nb[1], t->nb[1], nr, l.K, l.kp);
        return;
    }

    for (int64_t ib = 0; ib < l.K/AMX_QK; ib++) {
        int8_t      q[AMX_TILE][AMX_QK] = {};
        ggml_fp16_t s[2][AMX_TILE]      = {};
        ggml_fp16_t m[AMX_TILE]         = {};

        for (int r = 0; r < nr; r++) {
            float fs, fs1, fm;
            extract_block(t->type, src + (i01 + r)*t->nb[1], ib, q[r], fs, fs1, fm);
            s[0][r] = fp32_to_fp16(fs);
            s[1][r] = fp32_to_fp16(fs1);
            m[r]    = fp32_to_fp16(fm);
        }

        uint8_t * blk = (uint8_t *) dst + ib*l.block_size;
        memset(blk, 0, l.qsize);
        for (int g = 0; g < AMX_QK/4; g++) {
            for (int kk = 0; kk < 4; kk++) {
                const int k = 4*g + kk;
                if (kind_q8(l.knd)) {
                    for (int r = 0; r < AMX_TILE; r++) {
                        blk[128*g + 4*r + kk] = (uint8_t) q[r][k];
                    }
                } else {
                    for (int r = 0; r < 16; r++) {
                        blk[64*g + 4*r + kk] = (q[r][k] & 0xF) | ((q[r + 16][k] & 0xF) << 4);
                    }
                }
            }
            // element b of vector v is row 4*v + b/4, k value 4*g + b%4
            for (int h = 0; h < kind_nhi(l.knd); h++) {
                uint8_t * plane = blk + AMX_QNIB + h*AMX_QPLANE + 16*g;
                for (int b = 0; b < 16; b++) {
                    for (int v = 0; v < 8; v++) {
                        plane[b] |= ((q[4*v + b/4][4*g + b%4] >> (4 + h)) & 1) << v;
                    }
                }
            }
        }
        memcpy(blk + l.scale_offset(0), s[0], sizeof(s[0]));
        if (kind_split(l.knd)) {
            memcpy(blk + l.scale_offset(1), s[1], sizeof(s[1]));
        }
        if (kind_has_min(l.knd)) {
            memcpy(blk + l.min_offset(), m, sizeof(m));
        }
    }
}

static void pack_tensor(struct ggml_tensor * t, const void * data, const layout & l) {
    const int n_threads = (int) std::max<int64_t>(1, std::min<int64_t>(std::thread::hardware_concurrency(), l.n_panels/4));

    auto worker = [&](int ith) {
        for (int64_t ip = ith; ip < l.n_panels; ip += n_threads) {
            pack_panel(t, l, (const char *) data, ip);
        }
    };

    std::vector<std::thread> threads;
    for (int i = 1; i < n_threads; i++) {
        threads.emplace_back(worker, i);
    }
    worker(0);
    for (auto & th : threads) {
        th.join();
    }
}

//
// NEON kernels for small batches
//

// quantize a row of activations to int8 in blocks of AMX_QK, with the scale and the scaled sum of each block
static void quantize_row(const float * x, int64_t K, int8_t * qx, float * sx, float * sumx) {
    for (int64_t ib = 0; ib < K/AMX_QK; ib++) {
        const float * xb = x + ib*AMX_QK;

        float32x4_t v[8];
        float32x4_t amax = vdupq_n_f32(0.0f);
        for (int i = 0; i < 8; i++) {
            v[i] = vld1q_f32(xb + 4*i);
            amax = vmaxq_f32(amax, vabsq_f32(v[i]));
        }
        const float d  = vmaxvq_f32(amax)/127.0f;
        const float id = d != 0.0f ? 1.0f/d : 0.0f;

        int32x4_t qi[8];
        int32x4_t sum = vdupq_n_s32(0);
        for (int i = 0; i < 8; i++) {
            qi[i] = vcvtnq_s32_f32(vmulq_n_f32(v[i], id));
            sum   = vaddq_s32(sum, qi[i]);
        }
        for (int i = 0; i < 2; i++) {
            const int16x8_t lo = vcombine_s16(vmovn_s32(qi[4*i + 0]), vmovn_s32(qi[4*i + 1]));
            const int16x8_t hi = vcombine_s16(vmovn_s32(qi[4*i + 2]), vmovn_s32(qi[4*i + 3]));
            vst1q_s8(qx + ib*AMX_QK + 16*i, vcombine_s8(vmovn_s16(lo), vmovn_s16(hi)));
        }

        sx[ib]   = d;
        sumx[ib] = d*vaddvq_s32(sum);
    }
}

// convert a row of activations to f16, padded with zeros to kp
static void convert_row_f16(const float * x, int64_t K, int64_t kp, ggml_fp16_t * xh) {
    float16_t * o = (float16_t *) xh;
    int64_t k = 0;
    for (; k + 4 <= K; k += 4) {
        vst1_f16(o + k, vcvt_f16_f32(vld1q_f32(x + k)));
    }
    for (; k < K; k++) {
        o[k] = (float16_t) x[k];
    }
    for (; k < kp; k++) {
        o[k] = 0;
    }
}

// dst[j][r] = sum_k w[r][k]*x[j][k] for the NC activation rows x[j] and the rows r of a quantized panel
template <int NC, kind KIND>
static void gemv_q(const uint8_t * panel, const layout & l, const int8_t * const * qx, const float * const * sx, const float * const * sumx, float * const * dst, int nr) {
    float32x4_t facc[NC][8];
    for (int j = 0; j < NC; j++) {
        for (int v = 0; v < 8; v++) {
            facc[j][v] = vdupq_n_f32(0.0f);
        }
    }

    constexpr int  NHI     = kind_nhi(KIND);
    constexpr bool HAS_MIN = kind_has_min(KIND);
    constexpr bool SPLIT   = kind_split(KIND);
    constexpr int  OFFSET  = kind_offset(KIND);

    const uint8x16_t m4 = vdupq_n_u8(0xF);

    for (int64_t ib = 0; ib < l.K/AMX_QK; ib++) {
        const uint8_t * blk = panel + ib*l.block_size;

        // acc[j][v] holds rows [4*v, 4*v + 4)
        int32x4_t acc[NC][8];
        for (int j = 0; j < NC; j++) {
            for (int v = 0; v < 8; v++) {
                acc[j][v] = vdupq_n_s32(0);
            }
        }

        int8x16_t xv[NC];

#define AMX_GEMV_STEP(g)                                                                    \
        {                                                                                   \
            int8x16_t w[8];                                                                 \
            if constexpr (kind_q8(KIND)) {                                                  \
                for (int q = 0; q < 8; q++) {                                               \
                    w[q] = vld1q_s8((const int8_t *) blk + 128*(g) + 16*q);                 \
                }                                                                           \
            } else {                                                                        \
                uint8x16_t u[8];                                                            \
                for (int q = 0; q < 4; q++) {                                               \
                    const uint8x16_t b = vld1q_u8(blk + 64*(g) + 16*q);                     \
                    u[q + 0] = vandq_u8(b, m4);                                             \
                    u[q + 4] = vshrq_n_u8(b, 4);                                            \
                }                                                                           \
                for (int h = 0; h < NHI; h++) {                                             \
                    const uint8x16_t hb = vld1q_u8(blk + AMX_QNIB + h*AMX_QPLANE + 16*(g)); \
                    for (int v = 0; v < 8; v++) {                                           \
                        u[v] = vorrq_u8(u[v], vandq_u8(vshlq_u8(hb, vdupq_n_s8(4 + h - v)), \
                                                       vdupq_n_u8(0x10 << h)));             \
                    }                                                                       \
                }                                                                           \
                for (int v = 0; v < 8; v++) {                                               \
                    w[v] = vreinterpretq_s8_u8(u[v]);                                       \
                    if constexpr (OFFSET != 0) {                                            \
                        w[v] = vsubq_s8(w[v], vdupq_n_s8(OFFSET));                          \
                    }                                                                       \
                }                                                                           \
            }                                                                               \
            if ((g) % 4 == 0) {                                                             \
                for (int j = 0; j < NC; j++) {                                              \
                    xv[j] = vld1q_s8(qx[j] + ib*AMX_QK + 4*(g));                            \
                }                                                                           \
            }                                                                               \
            for (int j = 0; j < NC; j++) {                                                  \
                for (int v = 0; v < 8; v++) {                                               \
                    acc[j][v] = vdotq_laneq_s32(acc[j][v], w[v], xv[j], (g) % 4);           \
                }                                                                           \
            }                                                                               \
        }

        // add the dot products of acc, scaled by the 32 row scales at sp, to facc and reset acc
        auto flush = [&](const float16_t * sp) {
            for (int v = 0; v < 8; v++) {
                const float32x4_t s = vcvt_f32_f16(vld1_f16(sp + 4*v));
                for (int j = 0; j < NC; j++) {
                    facc[j][v] = vfmaq_n_f32(facc[j][v], vmulq_f32(vcvtq_f32_s32(acc[j][v]), s), sx[j][ib]);
                    acc[j][v]  = vdupq_n_s32(0);
                }
            }
        };

        AMX_GEMV_STEP(0) AMX_GEMV_STEP(1) AMX_GEMV_STEP(2) AMX_GEMV_STEP(3)
        if constexpr (SPLIT) {
            flush((const float16_t *) (blk + l.scale_offset(0)));
        }
        AMX_GEMV_STEP(4) AMX_GEMV_STEP(5) AMX_GEMV_STEP(6) AMX_GEMV_STEP(7)
        flush((const float16_t *) (blk + l.scale_offset(SPLIT ? 1 : 0)));

#undef AMX_GEMV_STEP

        // the mins multiply the sum of the activations of the block: Q4_0 has an implied min of -8*s
        if constexpr (HAS_MIN || kind_min8(KIND)) {
            const float16_t * sp = (const float16_t *) (blk + (HAS_MIN ? l.min_offset() : l.scale_offset(0)));
            for (int v = 0; v < 8; v++) {
                float32x4_t m = vcvt_f32_f16(vld1_f16(sp + 4*v));
                if constexpr (!HAS_MIN) {
                    m = vmulq_n_f32(m, -8.0f);
                }
                for (int j = 0; j < NC; j++) {
                    facc[j][v] = vfmaq_n_f32(facc[j][v], m, sumx[j][ib]);
                }
            }
        }
    }

    for (int j = 0; j < NC; j++) {
        if (nr == AMX_TILE) {
            for (int v = 0; v < 8; v++) {
                vst1q_f32(dst[j] + 4*v, facc[j][v]);
            }
        } else {
            float tmp[AMX_TILE];
            for (int v = 0; v < 8; v++) {
                vst1q_f32(tmp + 4*v, facc[j][v]);
            }
            memcpy(dst[j], tmp, nr*sizeof(float));
        }
    }
}

// dst[j][r] = sum_k w[r][k]*x[j][k] for the NC activation rows x[j] and the rows r of an f16 panel
// x is converted to f16 and padded with zeros to kp; the products are accumulated in f16 over 8 k values
template <int NC>
static void gemv_f16(const ggml_fp16_t * panel, int64_t kp, const ggml_fp16_t * const * x, float * const * dst, int nr) {
    const float16_t * p = (const float16_t *) panel;

    float32x4_t facc[NC][8];
    for (int j = 0; j < NC; j++) {
        for (int v = 0; v < 8; v++) {
            facc[j][v] = vdupq_n_f32(0.0f);
        }
    }

    for (int64_t k = 0; k < kp; k += 8) {
        float16x8_t xv[NC];
        float16x8_t acc[NC][4];
        for (int j = 0; j < NC; j++) {
            xv[j] = vld1q_f16((const float16_t *) x[j] + k);
            for (int i = 0; i < 4; i++) {
                acc[j][i] = vdupq_n_f16(0.0f);
            }
        }

#define AMX_GEMV_STEP(kk)                                                                   \
        {                                                                                   \
            float16x8_t w[4];                                                               \
            for (int i = 0; i < 4; i++) {                                                   \
                w[i] = vld1q_f16(p + (k + (kk))*AMX_TILE + 8*i);                            \
            }                                                                               \
            for (int j = 0; j < NC; j++) {                                                  \
                for (int i = 0; i < 4; i++) {                                               \
                    acc[j][i] = vfmaq_laneq_f16(acc[j][i], w[i], xv[j], (kk));              \
                }                                                                           \
            }                                                                               \
        }

        AMX_GEMV_STEP(0) AMX_GEMV_STEP(1) AMX_GEMV_STEP(2) AMX_GEMV_STEP(3)
        AMX_GEMV_STEP(4) AMX_GEMV_STEP(5) AMX_GEMV_STEP(6) AMX_GEMV_STEP(7)

#undef AMX_GEMV_STEP

        for (int j = 0; j < NC; j++) {
            for (int i = 0; i < 4; i++) {
                facc[j][2*i + 0] = vaddq_f32(facc[j][2*i + 0], vcvt_f32_f16(vget_low_f16(acc[j][i])));
                facc[j][2*i + 1] = vaddq_f32(facc[j][2*i + 1], vcvt_high_f32_f16(acc[j][i]));
            }
        }
    }

    for (int j = 0; j < NC; j++) {
        if (nr == AMX_TILE) {
            for (int v = 0; v < 8; v++) {
                vst1q_f32(dst[j] + 4*v, facc[j][v]);
            }
        } else {
            float tmp[AMX_TILE];
            for (int v = 0; v < 8; v++) {
                vst1q_f32(tmp + 4*v, facc[j][v]);
            }
            memcpy(dst[j], tmp, nr*sizeof(float));
        }
    }
}

//
// AMX path for large batches
//

// dequantize the blocks [ib0, ib1) of a quantized panel to an f16 panel [(ib1 - ib0)*AMX_QK][AMX_TILE]
template <kind KIND>
static void dequantize_panel(const uint8_t * panel, const layout & l, int64_t ib0, int64_t ib1, ggml_fp16_t * out) {
    constexpr int  NHI     = kind_nhi(KIND);
    constexpr bool HAS_MIN = kind_has_min(KIND);
    constexpr bool SPLIT   = kind_split(KIND);
    constexpr int  OFFSET  = kind_offset(KIND);

    float16_t * o = (float16_t *) out;

    // to gather the high bits of the rows r and r + 16 at k value kk of a group:
    // byte 4*(r%4) + kk of a plane, bits r/4 and r/4 + 4
    uint8x16_t gather[4];
    int8x16_t  shift_lo;
    int8x16_t  shift_hi;
    {
        uint8_t idx[4][16];
        int8_t  slo[16], shi[16];
        for (int r = 0; r < 16; r++) {
            for (int kk = 0; kk < 4; kk++) {
                idx[kk][r] = 4*(r%4) + kk;
            }
            slo[r] = 4 - r/4;
            shi[r] =   - r/4;
        }
        for (int kk = 0; kk < 4; kk++) {
            gather[kk] = vld1q_u8(idx[kk]);
        }
        shift_lo = vld1q_s8(slo);
        shift_hi = vld1q_s8(shi);
    }

    for (int64_t ib = ib0; ib < ib1; ib++) {
        const uint8_t * blk = panel + ib*l.block_size;

        // scales of each half of the block, and mins (Q4_0: -8*s)
        float16x8_t s[2][4];
        float16x8_t m[4];
        for (int i = 0; i < 4; i++) {
            s[0][i] = vld1q_f16((const float16_t *) (blk + l.scale_offset(0)) + 8*i);
            s[1][i] = SPLIT ? vld1q_f16((const float16_t *) (blk + l.scale_offset(1)) + 8*i) : s[0][i];
            if constexpr (HAS_MIN) {
                m[i] = vld1q_f16((const float16_t *) (blk + l.min_offset()) + 8*i);
            } else {
                m[i] = vmulq_n_f16(s[0][i], -8.0f);
            }
        }

        for (int g = 0; g < AMX_QK/4; g++) {
            const float16x8_t * sg = s[g < AMX_QK/8 ? 0 : 1];

            if constexpr (kind_q8(KIND)) {
                const int8x16x4_t b0 = vld4q_s8((const int8_t *) blk + 128*g);
                const int8x16x4_t b1 = vld4q_s8((const int8_t *) blk + 128*g + 64);
                for (int kk = 0; kk < 4; kk++) {
                    const int16x8_t q[4] = {
                        vmovl_s8(vget_low_s8(b0.val[kk])), vmovl_high_s8(b0.val[kk]),
                        vmovl_s8(vget_low_s8(b1.val[kk])), vmovl_high_s8(b1.val[kk]),
                    };
                    float16_t * row = o + ((ib - ib0)*AMX_QK + 4*g + kk)*AMX_TILE;
                    for (int i = 0; i < 4; i++) {
                        vst1q_f16(row + 8*i, vmulq_f16(vcvtq_f16_s16(q[i]), sg[i]));
                    }
                }
                continue;
            }

            const uint8x16x4_t b = vld4q_u8(blk + 64*g);
            uint8x16_t hb[2];
            for (int h = 0; h < NHI; h++) {
                hb[h] = vld1q_u8(blk + AMX_QNIB + h*AMX_QPLANE + 16*g);
            }

            for (int kk = 0; kk < 4; kk++) {
                // rows [0, 16) and [16, 32)
                uint8x16_t lo = vandq_u8(b.val[kk], vdupq_n_u8(0xF));
                uint8x16_t hi = vshrq_n_u8(b.val[kk], 4);
                for (int h = 0; h < NHI; h++) {
                    const uint8x16_t gb = vqtbl1q_u8(hb[h], gather[kk]);
                    const uint8x16_t mb = vdupq_n_u8(0x10 << h);
                    lo = vorrq_u8(lo, vandq_u8(vshlq_u8(gb, vaddq_s8(shift_lo, vdupq_n_s8(h))), mb));
                    hi = vorrq_u8(hi, vandq_u8(vshlq_u8(gb, vaddq_s8(shift_hi, vdupq_n_s8(h))), mb));
                }

                float16x8_t q[4];
                if constexpr (OFFSET != 0) {
                    const int8x16_t slo = vsubq_s8(vreinterpretq_s8_u8(lo), vdupq_n_s8(OFFSET));
                    const int8x16_t shi = vsubq_s8(vreinterpretq_s8_u8(hi), vdupq_n_s8(OFFSET));
                    q[0] = vcvtq_f16_s16(vmovl_s8(vget_low_s8(slo)));
                    q[1] = vcvtq_f16_s16(vmovl_high_s8(slo));
                    q[2] = vcvtq_f16_s16(vmovl_s8(vget_low_s8(shi)));
                    q[3] = vcvtq_f16_s16(vmovl_high_s8(shi));
                } else {
                    q[0] = vcvtq_f16_u16(vmovl_u8(vget_low_u8(lo)));
                    q[1] = vcvtq_f16_u16(vmovl_high_u8(lo));
                    q[2] = vcvtq_f16_u16(vmovl_u8(vget_low_u8(hi)));
                    q[3] = vcvtq_f16_u16(vmovl_high_u8(hi));
                }

                float16_t * row = o + ((ib - ib0)*AMX_QK + 4*g + kk)*AMX_TILE;
                for (int i = 0; i < 4; i++) {
                    if constexpr (HAS_MIN || kind_min8(KIND)) {
                        vst1q_f16(row + 8*i, vfmaq_f16(m[i], q[i], sg[i]));
                    } else {
                        vst1q_f16(row + 8*i, vmulq_f16(q[i], sg[i]));
                    }
                }
            }
        }
    }
}

// dispatch on the kind of a quantized panel
template <int NC>
static void gemv_q_any(const uint8_t * panel, const layout & l, const int8_t * const * qx, const float * const * sx, const float * const * sumx, float * const * dst, int nr) {
    switch (l.knd) {
        case KIND_Q4:  gemv_q<NC, KIND_Q4 >(panel, l, qx, sx, sumx, dst, nr); break;
        case KIND_Q4M: gemv_q<NC, KIND_Q4M>(panel, l, qx, sx, sumx, dst, nr); break;
        case KIND_Q5:  gemv_q<NC, KIND_Q5 >(panel, l, qx, sx, sumx, dst, nr); break;
        case KIND_Q5M: gemv_q<NC, KIND_Q5M>(panel, l, qx, sx, sumx, dst, nr); break;
        case KIND_Q6:  gemv_q<NC, KIND_Q6 >(panel, l, qx, sx, sumx, dst, nr); break;
        case KIND_Q8:  gemv_q<NC, KIND_Q8 >(panel, l, qx, sx, sumx, dst, nr); break;
    }
}

static void dequantize_panel_any(const uint8_t * panel, const layout & l, int64_t ib0, int64_t ib1, ggml_fp16_t * out) {
    switch (l.knd) {
        case KIND_Q4:  dequantize_panel<KIND_Q4 >(panel, l, ib0, ib1, out); break;
        case KIND_Q4M: dequantize_panel<KIND_Q4M>(panel, l, ib0, ib1, out); break;
        case KIND_Q5:  dequantize_panel<KIND_Q5 >(panel, l, ib0, ib1, out); break;
        case KIND_Q5M: dequantize_panel<KIND_Q5M>(panel, l, ib0, ib1, out); break;
        case KIND_Q6:  dequantize_panel<KIND_Q6 >(panel, l, ib0, ib1, out); break;
        case KIND_Q8:  dequantize_panel<KIND_Q8 >(panel, l, ib0, ib1, out); break;
    }
}

//
// MUL_MAT
//

struct mul_mat_plan {
    layout  l;
    int64_t N;          // src1 rows, flattened over dims 1-3
    bool    use_amx;

    // AMX path
    int64_t n_np;       // src1 panels
    int64_t n_nb;       // blocks of src1 panels
    int64_t kb_size;    // k values per K block
    int64_t n_kb;
    size_t  src1_size;  // packed src1
    size_t  wpanel_size;// per-thread dequantized src0 panel

    // NEON path
    size_t  qx_size;    // src1 quantized to int8, or converted to f16
    size_t  sx_size;    // src1 block scales and sums

    size_t  wsize;
};

#define AMX_N_BLOCK 16 // src1 panels per work item

static mul_mat_plan make_plan(const struct ggml_tensor * op, int n_threads) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];

    mul_mat_plan p = {};
    p.l       = get_layout(src0);
    p.N       = src1->ne[1]*src1->ne[2]*src1->ne[3];
    p.use_amx = p.N >= (p.l.fmt == FORMAT_F16 ? AMX_MIN_BATCH_F16 : AMX_MIN_BATCH_Q);

    const int64_t K = p.l.K;

    if (p.use_amx) {
        const int64_t kb_align = p.l.fmt == FORMAT_F16 ? 8 : AMX_QK;
        const int64_t kb_max   = std::max<int64_t>(AMX_K_BLOCK/sizeof(ggml_fp16_t)/kb_align*kb_align, kb_align);
        const int64_t kb_count = (K + kb_max - 1)/kb_max;

        p.kb_size     = ((K + kb_count - 1)/kb_count + kb_align - 1)/kb_align*kb_align;
        p.n_kb        = (K + p.kb_size - 1)/p.kb_size;
        p.n_np        = (p.N + AMX_TILE - 1)/AMX_TILE;
        p.n_nb        = (p.n_np + AMX_N_BLOCK - 1)/AMX_N_BLOCK;
        p.src1_size   = GGML_PAD(p.n_np*p.l.kp*AMX_TILE*sizeof(ggml_fp16_t), AMX_ALIGNMENT);
        p.wpanel_size = p.l.fmt == FORMAT_F16 ? 0 : GGML_PAD(p.kb_size*AMX_TILE*sizeof(ggml_fp16_t), AMX_ALIGNMENT);
        p.wsize       = AMX_ALIGNMENT + p.src1_size + n_threads*p.wpanel_size;
    } else if (p.l.fmt == FORMAT_F16) {
        p.qx_size = GGML_PAD(p.N*p.l.kp*sizeof(ggml_fp16_t), 64);
        p.wsize   = 64 + p.qx_size;
    } else {
        p.qx_size = GGML_PAD(p.N*K, 64);
        p.sx_size = GGML_PAD(p.N*(K/AMX_QK)*sizeof(float), 64);
        p.wsize   = 64 + p.qx_size + 2*p.sx_size;
    }

    return p;
}

// compute the rows of dst of the panel ip of src0
static void mul_mat_neon_panel(const struct ggml_tensor * op, const mul_mat_plan & p, int64_t ip, const int8_t * qx, const float * sx, const float * sumx) {
    const struct ggml_tensor * src0 = op->src[0];

    const layout & l = p.l;
    const int64_t  K = l.K;

    const size_t nb1 = op->nb[1];

    const uint8_t * panel = (const uint8_t *) src0->data + ip*l.panel_size;
    const int       nr    = (int) std::min<int64_t>(AMX_TILE, l.M - ip*AMX_TILE);

    for (int64_t i = 0; i < p.N; ) {
        const int nc = (int) std::min<int64_t>(2, p.N - i);

        float * d[2];
        for (int j = 0; j < nc; j++) {
            d[j] = (float *) ((char *) op->data + (i + j)*nb1) + ip*AMX_TILE;
        }

        if (l.fmt == FORMAT_F16) {
            const ggml_fp16_t * x[2];
            for (int j = 0; j < nc; j++) {
                x[j] = (const ggml_fp16_t *) qx + (i + j)*l.kp;
            }
            if (nc == 2) {
                gemv_f16<2>((const ggml_fp16_t *) panel, l.kp, x, d, nr);
            } else {
                gemv_f16<1>((const ggml_fp16_t *) panel, l.kp, x, d, nr);
            }
        } else {
            const int8_t * q[2];
            const float  * s[2];
            const float  * sm[2];
            for (int j = 0; j < nc; j++) {
                q[j]  = qx   + (i + j)*K;
                s[j]  = sx   + (i + j)*(K/AMX_QK);
                sm[j] = sumx + (i + j)*(K/AMX_QK);
            }

            if (nc == 2) {
                gemv_q_any<2>(panel, l, q, s, sm, d, nr);
            } else {
                gemv_q_any<1>(panel, l, q, s, sm, d, nr);
            }
        }

        i += nc;
    }
}

static void mul_mat_neon(struct ggml_compute_params * params, struct ggml_tensor * op, const mul_mat_plan & p) {
    const struct ggml_tensor * src1 = op->src[1];

    const layout & l = p.l;
    const int64_t  K = l.K;

    const int ith = params->ith;
    const int nth = params->nth;

    // src1 and dst rows are uniformly strided over the flattened dims 1-3
    const size_t nb11 = src1->nb[1];

    char   * wdata = (char *) GGML_PAD((uintptr_t) params->wdata, 64);
    int8_t * qx    = (int8_t *) wdata;
    float  * sx    = (float  *) (wdata + p.qx_size);
    float  * sumx  = (float  *) (wdata + p.qx_size + p.sx_size);

    for (int64_t i = ith; i < p.N; i += nth) {
        const float * x = (const float *) ((const char *) src1->data + i*nb11);
        if (l.fmt == FORMAT_F16) {
            convert_row_f16(x, K, l.kp, (ggml_fp16_t *) qx + i*l.kp);
        } else {
            quantize_row(x, K, qx + i*K, sx + i*(K/AMX_QK), sumx + i*(K/AMX_QK));
        }
    }

    // panels are distributed in chunks on demand, since the threads may run on cores of different speeds
    const int64_t chunk_size = std::max<int64_t>(1, l.n_panels/(4*nth));
    const int64_t n_chunks   = (l.n_panels + chunk_size - 1)/chunk_size;

    if (ith == 0) {
        ggml_threadpool_chunk_set(params->threadpool, nth);
    }
    ggml_barrier(params->threadpool);

    for (int64_t chunk = ith; chunk < n_chunks; chunk = ggml_threadpool_chunk_add(params->threadpool, 1)) {
        const int64_t ip1 = std::min(l.n_panels, (chunk + 1)*chunk_size);
        for (int64_t ip = chunk*chunk_size; ip < ip1; ip++) {
            mul_mat_neon_panel(op, p, ip, qx, sx, sumx);
        }
    }
}

static void mul_mat_amx(struct ggml_compute_params * params, struct ggml_tensor * op, const mul_mat_plan & p) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];

    const layout & l = p.l;
    const int64_t  K = l.K;

    const int ith = params->ith;
    const int nth = params->nth;

    const size_t nb11 = src1->nb[1];
    const size_t nb1  = op->nb[1];

    char        * wdata  = (char *) GGML_PAD((uintptr_t) params->wdata, AMX_ALIGNMENT);
    ggml_fp16_t * psrc1  = (ggml_fp16_t *) wdata;
    ggml_fp16_t * wpanel = (ggml_fp16_t *) (wdata + p.src1_size + ith*p.wpanel_size);

    const size_t src1_panel = l.kp*AMX_TILE; // f16 elements per packed src1 panel

    // pack src1
    for (int64_t inp = ith; inp < p.n_np; inp += nth) {
        const int64_t i11 = inp*AMX_TILE;
        const int     nr  = (int) std::min<int64_t>(AMX_TILE, p.N - i11);
        amx_pack_f32_f16(psrc1 + inp*src1_panel, (const char *) src1->data + i11*nb11, nb11, nr, K, l.kp);
    }

    if (ith == 0) {
        ggml_threadpool_chunk_set(params->threadpool, nth);
    }
    ggml_barrier(params->threadpool);

    // ordered so that concurrent work items share the same block of src1 panels
    const int64_t n_items = p.n_nb*l.n_panels;

    AMX_SET();

    for (int64_t item = ith; item < n_items; item = ggml_threadpool_chunk_add(params->threadpool, 1)) {
        const int64_t imp = item%l.n_panels;
        const int64_t inb = item/l.n_panels;
        const int64_t i01 = imp*AMX_TILE;
        const int     nc  = (int) std::min<int64_t>(AMX_TILE, l.M - i01);

        const uint8_t * panel = (const uint8_t *) src0->data + imp*l.panel_size;

        for (int64_t ikb = 0; ikb < p.n_kb; ikb++) {
            const int64_t k0  = ikb*p.kb_size;
            const int64_t knp = std::min(p.kb_size, l.kp - k0);

            const ggml_fp16_t * a;
            if (l.fmt == FORMAT_F16) {
                a = (const ggml_fp16_t *) panel + k0*AMX_TILE;
            } else {
                const int64_t ib0 = k0/AMX_QK;
                const int64_t ib1 = ib0 + knp/AMX_QK;
                dequantize_panel_any(panel, l, ib0, ib1, wpanel);
                a = wpanel;
            }

            const int64_t inp0 = inb*AMX_N_BLOCK;
            const int64_t inp1 = std::min(p.n_np, inp0 + AMX_N_BLOCK);
            for (int64_t inp = inp0; inp < inp1; inp++) {
                const int64_t i11 = inp*AMX_TILE;
                const int     nr  = (int) std::min<int64_t>(AMX_TILE, p.N - i11);

                const ggml_fp16_t * b = psrc1 + inp*src1_panel + k0*AMX_TILE;
                float             * c = (float *) ((char *) op->data + i11*nb1) + i01;

                amx_kernel_f16(a, b, c, nb1/sizeof(float), knp, nr, nc, ikb > 0);
            }
        }
    }

    AMX_CLR();
}

class tensor_traits : public ggml::cpu::tensor_traits {
    bool work_size(int n_threads, const struct ggml_tensor * op, size_t & size) override {
        size = make_plan(op, n_threads).wsize;
        return true;
    }

    bool compute_forward(struct ggml_compute_params * params, struct ggml_tensor * op) override {
        if (op->op != GGML_OP_MUL_MAT) {
            return false;
        }

        const mul_mat_plan p = make_plan(op, params->nth);
        GGML_ASSERT(params->wsize >= p.wsize);

        if (p.use_amx) {
            mul_mat_amx(params, op, p);
        } else {
            mul_mat_neon(params, op, p);
        }
        return true;
    }
};

static tensor_traits * get_tensor_traits() {
    static tensor_traits traits;
    return &traits;
}

//
// buffer
//

static void buffer_free_buffer(ggml_backend_buffer_t buffer) {
    free(buffer->context);
}

static void * buffer_get_base(ggml_backend_buffer_t buffer) {
    return buffer->context;
}

static enum ggml_status buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    tensor->extra = (void *) get_tensor_traits();
    return GGML_STATUS_SUCCESS;

    GGML_UNUSED(buffer);
}

static void buffer_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    memset((char *) tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

static void buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    const layout l = get_layout(tensor);
    if (l.fmt == FORMAT_NONE) {
        memcpy((char *) tensor->data + offset, data, size);
        return;
    }

    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    pack_tensor(tensor, data, l);

    GGML_UNUSED(buffer);
}

static void buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    // packed tensors cannot be read back
    GGML_ASSERT(get_layout(tensor).fmt == FORMAT_NONE);
    memcpy(data, (const char *) tensor->data + offset, size);

    GGML_UNUSED(buffer);
}

static void buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    memset(buffer->context, value, buffer->size);
}

static const ggml_backend_buffer_i buffer_interface = {
    /* .free_buffer     = */ buffer_free_buffer,
    /* .get_base        = */ buffer_get_base,
    /* .init_tensor     = */ buffer_init_tensor,
    /* .memset_tensor   = */ buffer_memset_tensor,
    /* .set_tensor      = */ buffer_set_tensor,
    /* .get_tensor      = */ buffer_get_tensor,
    /* .set_tensor_2d   = */ nullptr,
    /* .get_tensor_2d   = */ nullptr,
    /* .cpy_tensor      = */ nullptr,
    /* .clear           = */ buffer_clear,
    /* .reset           = */ nullptr,
};

//
// buffer type
//

static const char * buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return "CPU_APPLEAMX";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * data = aligned_alloc(AMX_ALIGNMENT, GGML_PAD(std::max<size_t>(size, 1), AMX_ALIGNMENT));
    if (data == nullptr) {
        GGML_LOG_ERROR("%s: failed to allocate buffer of size %zu\n", __func__, size);
        return nullptr;
    }

    return ggml_backend_buffer_init(buft, buffer_interface, data, size);
}

static size_t buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return AMX_ALIGNMENT;

    GGML_UNUSED(buft);
}

static size_t buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {
    const layout l = get_layout(tensor);
    return std::max(ggml_nbytes(tensor), l.size());

    GGML_UNUSED(buft);
}

class extra_buffer_type : ggml::cpu::extra_buffer_type {
    bool supports_op(ggml_backend_dev_t, const struct ggml_tensor * op) override {
        if (op->op != GGML_OP_MUL_MAT) {
            return false;
        }

        const struct ggml_tensor * src0 = op->src[0];
        const struct ggml_tensor * src1 = op->src[1];

        if (!src0->buffer || src0->buffer->buft != ggml_backend_cpu_appleamx_buffer_type() ||
            src0->view_src || get_layout(src0).fmt == FORMAT_NONE) {
            return false;
        }
        if (src1->buffer && !ggml_backend_buft_is_host(src1->buffer->buft)) {
            return false;
        }

        // src1 and dst rows must be uniformly strided over the flattened dims 1-3
        return src1->type == GGML_TYPE_F32 &&
               op->type   == GGML_TYPE_F32 &&
               src1->nb[0] == sizeof(float) &&
               op->nb[0]   == sizeof(float) &&
               src1->nb[2] == src1->ne[1]*src1->nb[1] && src1->nb[3] == src1->ne[2]*src1->nb[2] &&
               op->nb[2]   == op->ne[1]*op->nb[1]     && op->nb[3]   == op->ne[2]*op->nb[2];
    }

    ggml::cpu::tensor_traits * get_tensor_traits(const struct ggml_tensor * op) override {
        if (op->op == GGML_OP_MUL_MAT && op->src[0]->buffer &&
            op->src[0]->buffer->buft == ggml_backend_cpu_appleamx_buffer_type()) {
            return (ggml::cpu::tensor_traits *) op->src[0]->extra;
        }
        return nullptr;
    }
};

}  // namespace ggml::cpu::appleamx

ggml_backend_buffer_type_t ggml_backend_cpu_appleamx_buffer_type(void) {
    using namespace ggml::cpu::appleamx;

    static struct ggml_backend_buffer_type buffer_type = {
        /* .iface = */ {
            /* .get_name         = */ buffer_type_get_name,
            /* .alloc_buffer     = */ buffer_type_alloc_buffer,
            /* .alloc_buffer_n   = */ nullptr,
            /* .get_alignment    = */ buffer_type_get_alignment,
            /* .get_max_size     = */ nullptr, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ buffer_type_get_alloc_size,
            /* .get_alloc_size_n = */ nullptr,
            /* .is_host          = */ nullptr,
        },
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ new ggml::cpu::appleamx::extra_buffer_type(),
    };

    return &buffer_type;
}
