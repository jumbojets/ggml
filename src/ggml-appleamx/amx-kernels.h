// Apple AMX packing helpers and matrix micro kernels
//
// shared by the AppleAMX backend and the Apple AMX buffer type of the CPU backend
//
// The output of a MUL_MAT is computed in 32x32 tiles as a sum of outer products:
//   dst[i1][i0] = sum_k src0[i0][k] * src1[i1][k]
// Both operands are packed into k-major panels of 32 rows, so that for each k
// the 32 values of a panel are contiguous and can be loaded into an AMX X or Y
// register. src0 panels go into X (Z columns, contiguous in dst) and src1
// panels go into Y (Z rows).
//
// Threads that use these kernels must enable AMX with AMX_SET() first and
// disable it with AMX_CLR() when they are done.

#pragma once

#include "ggml.h"
#include "amx.h"

#include <arm_neon.h>

#include <cstdint>
#include <cstring>

#define AMX_TILE      32  // rows of a packed panel, size of an output tile
#define AMX_ALIGNMENT 128 // pair loads need 128-byte alignment

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
static inline void amx_pack_pad(T * dst, int nr, int64_t K, int64_t kp) {
    if (nr < AMX_TILE) {
        for (int64_t k = 0; k < K; k++) {
            memset(dst + k*AMX_TILE + nr, 0, (AMX_TILE - nr)*sizeof(T));
        }
    }
    memset(dst + K*AMX_TILE, 0, (kp - K)*AMX_TILE*sizeof(T));
}

// f32 rows -> f32 panel
static inline void amx_pack_f32_f32(float * dst, const char * src, size_t nb, int nr, int64_t K, int64_t kp) {
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
static inline void amx_pack_f32_f16(ggml_fp16_t * dst_, const char * src, size_t nb, int nr, int64_t K, int64_t kp) {
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
static inline void amx_pack_f16_f16(ggml_fp16_t * dst, const char * src, size_t nb, int nr, int64_t K, int64_t kp) {
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

static inline void amx_kernel_f32(const float * a, const float * b, float * c, size_t ldc, int64_t kp, int nr, int nc, bool acc) {
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

static inline void amx_kernel_f16(const ggml_fp16_t * a, const ggml_fp16_t * b, float * c, size_t ldc, int64_t kp, int nr, int nc, bool acc) {
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
