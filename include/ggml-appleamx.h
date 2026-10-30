#pragma once

#include "ggml.h"
#include "ggml-backend.h"


#ifdef  __cplusplus
extern "C" {
#endif

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_appleamx_init(void);

GGML_BACKEND_API bool ggml_backend_is_appleamx(ggml_backend_t backend);

// number of threads used to pack the operands and to drive the AMX units
// each CPU cluster has its own AMX unit, so this should cover all of the performance cores
GGML_BACKEND_API void ggml_backend_appleamx_set_n_threads(ggml_backend_t backend_appleamx, int n_threads);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_appleamx_reg(void);


#ifdef  __cplusplus
}
#endif
