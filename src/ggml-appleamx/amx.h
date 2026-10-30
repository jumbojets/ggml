// Apple AMX instruction macros.
//
// From https://github.com/jumbojets/exo-apple-amx/blob/master/appleamx/amx.h
// (MIT License, Copyright (c) 2025-2026 James Taylor), which is in turn
// modified from https://github.com/corsix/amx/blob/main/aarch64.h
// (MIT License, Copyright (c) 2022 Peter Cawley).
// Instruction and operand encodings are documented in the corsix/amx repository.

#pragma once
#include <stdint.h>

#define AMX_NOP_OP_IMM5(op, imm5) \
  __asm("nop\nnop\nnop\n.word (0x201000 + (%0 << 5) + %1)" : : "i"(op), "i"(imm5) : "memory")

#define AMX_OP_GPR(op, gpr) \
  __asm(".word (0x201000 + (%0 << 5) + 0%1 - ((0%1 >> 4) * 6))" : : "i"(op), "r"((uint64_t)(gpr)) : "memory")

// reg is a register index; the pointer need not be aligned for single-register moves
#define AMX_LDST(op, ptr, reg, flags) \
  AMX_OP_GPR(op, ((uint64_t)&*(ptr)) | ((uint64_t)(reg) << 56) | (flags))

// y and x are byte offsets into the Y and X register files, z is a Z row index
#define AMX_ALU(op, y, x, z, flags) \
  AMX_OP_GPR(op, ((uint64_t)(y)) | ((uint64_t)(x) << 10) | ((uint64_t)(z) << 20) | (flags))

#define AMX_LDX(ptr, reg, flags)   AMX_LDST(0, ptr, reg, flags)
#define AMX_LDY(ptr, reg, flags)   AMX_LDST(1, ptr, reg, flags)
#define AMX_STX(ptr, reg, flags)   AMX_LDST(2, ptr, reg, flags)
#define AMX_STY(ptr, reg, flags)   AMX_LDST(3, ptr, reg, flags)
#define AMX_LDZ(ptr, reg, flags)   AMX_LDST(4, ptr, reg, flags)
#define AMX_STZ(ptr, reg, flags)   AMX_LDST(5, ptr, reg, flags)
#define AMX_LDZI(ptr, reg, flags)  AMX_LDST(6, ptr, reg, flags)
#define AMX_STZI(ptr, reg, flags)  AMX_LDST(7, ptr, reg, flags)
#define AMX_EXTRX(gpr)             AMX_OP_GPR(8, gpr)
#define AMX_EXTRY(gpr)             AMX_OP_GPR(9, gpr)
#define AMX_FMA64(y, x, z, flags)  AMX_ALU(10, y, x, z, flags)
#define AMX_FMS64(y, x, z, flags)  AMX_ALU(11, y, x, z, flags)
#define AMX_FMA32(y, x, z, flags)  AMX_ALU(12, y, x, z, flags)
#define AMX_FMS32(y, x, z, flags)  AMX_ALU(13, y, x, z, flags)
#define AMX_MAC16(y, x, z, flags)  AMX_ALU(14, y, x, z, flags)
#define AMX_FMA16(y, x, z, flags)  AMX_ALU(15, y, x, z, flags)
#define AMX_FMS16(y, x, z, flags)  AMX_ALU(16, y, x, z, flags)
#define AMX_SET()                  AMX_NOP_OP_IMM5(17, 0)
#define AMX_CLR()                  AMX_NOP_OP_IMM5(17, 1)
#define AMX_VECINT(y, x, z, flags) AMX_ALU(18, y, x, z, flags)
#define AMX_VECFP(y, x, z, flags)  AMX_ALU(19, y, x, z, flags)
#define AMX_MATINT(y, x, z, flags) AMX_ALU(20, y, x, z, flags)
#define AMX_MATFP(y, x, z, flags)  AMX_ALU(21, y, x, z, flags)
#define AMX_GENLUT(src, flags)     AMX_OP_GPR(22, ((uint64_t)(src)) | (flags))

// Load / store flags. A pair move needs a 128-byte aligned pointer.
#define AMX_LDST_PAIR (1ull << 62)
// ldzi / stzi move 64 bytes between memory and one half of a Z register pair:
// even 32-bit lanes go to the even register, odd lanes to the odd one. The
// register field is (pair << 1) | half, i.e. the even row index plus the half.

// fma16 / fms16 matrix mode: accumulate into f32. Z is then one 32x32 f32 grid
// over all 64 rows, logical row j in registers (2j, 2j+1) as ldzi/stzi lay it out.
#define AMX_Z_F32 (1ull << 62)

// fma* / fms* / mac16 flags
#define AMX_VECTOR (1ull << 63)  // pointwise z[i] += x[i]*y[i] instead of the outer product
#define AMX_SKIP_X (1ull << 29)
#define AMX_SKIP_Y (1ull << 28)
#define AMX_SKIP_Z (1ull << 27)
// Restrict to the first n lanes of X (columns of Z) or Y (rows of Z).
// The 5-bit field wraps, so a full lane count encodes as 0, which means all lanes.
#define AMX_ENABLE_X_FIRST(n) ((2ull << 46) | (((uint64_t)(n) & 31) << 41))
#define AMX_ENABLE_Y_FIRST(n) ((2ull << 37) | (((uint64_t)(n) & 31) << 32))

#define AMX_EXTRX_FROM_Y(xreg, yreg) \
  AMX_EXTRX((1ull << 27) | ((uint64_t)(yreg) << 20) | ((uint64_t)(xreg) << 16))
#define AMX_EXTRY_FROM_X(yreg, xreg) \
  AMX_EXTRY((1ull << 27) | ((uint64_t)(xreg) << 20) | ((uint64_t)(yreg) << 6))
// lane: 0 = 64-bit, 1 = 32-bit, 2 = 16-bit; for extrh only the write mask, for extrv also the Z cell size
#define AMX_EXTRH(xreg, zrow, lane) \
  AMX_EXTRX(((uint64_t)(lane) << 28) | ((uint64_t)(zrow) << 20) | (((uint64_t)(xreg) * 64) << 10))
// zcol: the accumulator's base register + column * element bytes (row i of the accumulator is register i * bytes + base)
#define AMX_EXTRV(yreg, zcol, lane) \
  AMX_EXTRY(((uint64_t)(lane) << 28) | ((uint64_t)(zcol) << 20) | ((uint64_t)(yreg) * 64))

// matint mode 8, lane width 10: an i8 outer product accumulated into i32
#define AMX_MATINT_MAC8_I32 ((8ull << 47) | (10ull << 42))
#define AMX_MATINT_X_SIGNED (1ull << 63)
#define AMX_MATINT_Y_SIGNED (1ull << 26)
#define AMX_MATINT_ZERO_Z (3ull << 32)  // write 0 instead of the result

// ldz / stz of one 256-byte row of the mode 8 tile
static inline void amx_ldzq(const int32_t *src, uint64_t reg) {
  int32_t t[4][16];
  for (int j = 0; j < 64; j++) t[j % 4][j / 4] = src[j];
  for (int q = 0; q < 4; q++) AMX_LDZ(t[q], reg + q, 0);
}
static inline void amx_stzq(int32_t *dst, uint64_t reg) {
  int32_t t[4][16];
  for (int q = 0; q < 4; q++) AMX_STZ(t[q], reg + q, 0);
  for (int j = 0; j < 64; j++) dst[j] = t[j % 4][j / 4];
}
