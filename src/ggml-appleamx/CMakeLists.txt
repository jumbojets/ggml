if (NOT APPLE OR NOT CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    message(FATAL_ERROR "The Apple AMX backend requires Apple silicon")
endif()

message(STATUS "Apple AMX found")

ggml_add_backend_library(ggml-appleamx
                         ggml-appleamx.cpp
                        )
