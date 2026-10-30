# ggml

<div align="center">

<img src="https://raw.githubusercontent.com/ggml-org/media/refs/heads/master/logo/ggml-logo.jpg" width="256" height="256" alt="ggml logo" />

<b>Tensor library for machine learning</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/ggml?filter=v*)](https://github.com/ggml-org/ggml/releases)
[![CI](https://github.com/ggml-org/ggml/actions/workflows/build-cpu.yml/badge.svg)](https://github.com/ggml-org/ggml/actions/workflows/build-cpu.yml)

</div>

## Apple AMX fork

This is a fork of [ggml-org/ggml](https://github.com/ggml-org/ggml). The branch
[`claude/ggml-apple-amx-backend-05d157`](https://github.com/jumbojets/ggml/tree/claude/ggml-apple-amx-backend-05d157)
adds support for the undocumented Apple AMX matrix coprocessor found in M1-M3 Apple silicon. The instruction
macros come from [jumbojets/exo-apple-amx](https://github.com/jumbojets/exo-apple-amx), based on
[corsix/amx](https://github.com/corsix/amx).

It adds two components, both enabled with `-DGGML_APPLEAMX=ON`:

- **`CPU_APPLEAMX` buffer type of the CPU backend** ([src/ggml-cpu/apple-amx](src/ggml-cpu/apple-amx)): `MUL_MAT`
  weights are packed once, when they are loaded, into panels of 32 rows. Large batches dequantize a panel to f16
  and run on AMX; small batches, including token generation, use NEON dot products on the packed layout.
  Supported weight types: F16, Q4_0, Q4_1, Q4_K, Q8_0. llama.cpp uses it automatically (`--repack 1`, the default).
- **`AppleAMX` backend** ([src/ggml-appleamx](src/ggml-appleamx)): an accelerator backend, like BLAS, that runs
  large `MUL_MAT` ops on AMX for weights in plain host buffers (any type with a `to_float`), packing the operands
  on every call. The minimum matrix size is set with `GGML_APPLEAMX_MIN_BATCH` (default 32).

Each CPU cluster has its own AMX unit, so the work is spread over several threads (the performance cores by
default).

### Build

```bash
git clone -b claude/ggml-apple-amx-backend-05d157 https://github.com/jumbojets/ggml
cd ggml

cmake -B build -DGGML_APPLEAMX=ON
cmake --build build --config Release -j 8

# tests
./build/bin/test-appleamx                                         # packed weights vs plain CPU weights
./build/bin/test-backend-ops test -b AppleAMX -o MUL_MAT          # AppleAMX backend vs CPU
```

To use it with llama.cpp, replace the `ggml` directory of llama.cpp with this tree (the branch is based on the
llama.cpp sync at commit `e486a4b5`) and configure llama.cpp with `-DGGML_APPLEAMX=ON`.

### Performance

Llama-3.2-1B-Instruct with `llama-bench` on an M1 Max (AC power, 6 threads, 5 repetitions), in tokens/s:

| Model | Test  |  CPU | Accelerate (BLAS) | AppleAMX backend | **CPU_APPLEAMX** | Metal (GPU) |
|-------|-------|-----:|------------------:|-----------------:|-----------------:|------------:|
| F16   | pp32  |  405 |               110 |              320 |          **853** |        1850 |
|       | pp512 |  449 |               628 |              904 |         **1111** |        4005 |
|       | tg128 | 41.3 |              41.5 |             40.8 |             40.4 |         117 |
| Q8_0  | pp32  |  584 |               132 |              234 |          **864** |        1876 |
|       | pp512 |  603 |               630 |              890 |         **1129** |        3800 |
|       | tg128 | 66.8 |              68.0 |             71.7 |         **75.7** |         187 |
| Q4_0  | pp32  |  608 |               114 |              225 |          **853** |        2077 |
|       | pp512 |  614 |               658 |              899 |         **1089** |        3887 |
|       | tg128 |  112 |               101 |              100 |              107 |         269 |

CPU uses the CPU backend's own weight repacking; the Accelerate and AppleAMX backend columns use `--repack 0`,
since those backends cannot read repacked weights.

### Limitations

- Only tested on an M1 Max. AMX is undocumented, and it is unknown whether M4 and later still support it.
- `CPU_APPLEAMX` does not pack Q5/Q6 K-quants and the other quantization types yet; those weights stay on the CPU.
- Packed weights cannot be read back from the buffer.
- The AppleAMX backend handles only `MUL_MAT` (no `MUL_MAT_ID` or `OUT_PROD`).

## Quick start

Build from source:

```bash
git clone https://github.com/ggml-org/ggml
cd ggml

mkdir build && cd build
cmake ..
cmake --build . --config Release -j 8
```

For a minimal, fully commented example (matrix multiplication), see [examples/simple](examples/simple).

## Description

The main goal of `ggml` is to be a simple, portable, and efficient tensor library for machine learning with minimal setup.

- Plain C/C++ implementation without any dependencies
- Cross-platform - x86, ARM, RISC-V, LoongArch, PowerPC, s390x, and WebAssembly
- SIMD-optimized kernels for x86, ARM, and RISC-V
- Broad backend support - CPU, GPU, NPU, and browser
- 2- to 8-bit integer quantization, plus MXFP4 and NVFP4 microscaling formats
- Zero memory allocations during runtime

## Documentation

- [The GGUF file format](docs/gguf.md)
- [Introduction to ggml](https://huggingface.co/blog/introduction-to-ggml)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)

## Contributing

- For changes to the core `ggml` library (including to the CMake build system), please open a PR in [llama.cpp](https://github.com/ggml-org/llama.cpp) - doing so will make your PR more visible, better tested, and more likely to be reviewed
