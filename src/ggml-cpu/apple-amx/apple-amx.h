#pragma once

#include "ggml-backend.h"

// buffer type of the CPU backend that stores MUL_MAT weights pre-packed for the Apple AMX coprocessor
ggml_backend_buffer_type_t ggml_backend_cpu_appleamx_buffer_type(void);
