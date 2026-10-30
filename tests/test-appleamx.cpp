// Test of the CPU_APPLEAMX buffer type of the CPU backend
//
// MUL_MAT with weights pre-packed in the CPU_APPLEAMX buffer type is compared against
// the same weights in a plain CPU buffer and an f64 reference of the dequantized weights.
//
// usage: test-appleamx [perf]
//   perf: compare the speed of both instead, over a range of batch sizes
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <random>
#include <vector>
#include <chrono>

static ggml_backend_buffer_type_t find_extra_buft(const char * name) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    auto fn = (ggml_backend_dev_get_extra_bufts_t) ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev), "ggml_backend_dev_get_extra_bufts");
    for (auto * b = fn(dev); b && *b; b++) {
        if (strcmp(ggml_backend_buft_name(*b), name) == 0) return *b;
    }
    return nullptr;
}

struct result { double nmse_amx, nmse_cpu, ms_amx, ms_cpu; };

static result run(ggml_backend_t cpu, ggml_backend_buffer_type_t buft, ggml_type type, int64_t M, int64_t K, int64_t N, int64_t N2, int reps) {
    std::mt19937 rng(1234);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    ggml_init_params ip = { ggml_tensor_overhead()*16, nullptr, true };
    ggml_context * ctx_w   = ggml_init(ip);
    ggml_context * ctx_cpu = ggml_init(ip);

    ggml_tensor * w  = ggml_new_tensor_2d(ctx_w,   type, K, M);
    ggml_tensor * w2 = ggml_new_tensor_2d(ctx_cpu, type, K, M);
    ggml_tensor * x  = ggml_new_tensor_3d(ctx_cpu, GGML_TYPE_F32, K, N, N2);
    ggml_tensor * y1 = ggml_mul_mat(ctx_cpu, w,  x);
    ggml_tensor * y2 = ggml_mul_mat(ctx_cpu, w2, x);

    ggml_backend_buffer_t buf_w   = ggml_backend_alloc_ctx_tensors_from_buft(ctx_w, buft);
    ggml_backend_buffer_t buf_cpu = ggml_backend_alloc_ctx_tensors_from_buft(ctx_cpu, ggml_backend_cpu_buffer_type());

    std::vector<float> wf(M*K), xf(K*N*N2);
    for (auto & v : wf) v = dist(rng)*0.1f;
    for (auto & v : xf) v = dist(rng);
    std::vector<char> wq(ggml_nbytes(w));
    ggml_quantize_chunk(type, wf.data(), wq.data(), 0, M, K, nullptr);
    ggml_backend_tensor_set(w,  wq.data(), 0, wq.size());
    ggml_backend_tensor_set(w2, wq.data(), 0, wq.size());
    ggml_backend_tensor_set(x,  xf.data(), 0, xf.size()*sizeof(float));

    // reference from the dequantized weights
    std::vector<float> wd(M*K);
    ggml_get_type_traits(type)->to_float(wq.data(), wd.data(), M*K);
    std::vector<double> ref(M*N*N2);
    for (int64_t n = 0; n < N*N2; n++) {
        for (int64_t m = 0; m < M; m++) {
            double s = 0;
            for (int64_t k = 0; k < K; k++) s += (double) wd[m*K + k]*xf[n*K + k];
            ref[n*M + m] = s;
        }
    }

    ggml_context * ctx_g = ggml_init({ 2*ggml_graph_overhead(), nullptr, true });
    result r = {};
    for (int which = 0; which < 2; which++) {
        ggml_tensor * y = which == 0 ? y1 : y2;
        ggml_cgraph * gf = ggml_new_graph(ctx_g);
        ggml_build_forward_expand(gf, y);
        ggml_backend_graph_compute(cpu, gf);
        double ms = 1e9;
        for (int i = 0; i < reps; i++) {
            auto t0 = std::chrono::steady_clock::now();
            ggml_backend_graph_compute(cpu, gf);
            ms = std::min(ms, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }
        std::vector<float> out(M*N*N2);
        ggml_backend_tensor_get(y, out.data(), 0, out.size()*sizeof(float));
        double num = 0, den = 0;
        for (size_t i = 0; i < out.size(); i++) { num += (out[i] - ref[i])*(out[i] - ref[i]); den += ref[i]*ref[i]; }
        (which == 0 ? r.nmse_amx : r.nmse_cpu) = num/den;
        (which == 0 ? r.ms_amx : r.ms_cpu) = ms;
    }

    ggml_free(ctx_g);
    ggml_backend_buffer_free(buf_w);
    ggml_backend_buffer_free(buf_cpu);
    ggml_free(ctx_w);
    ggml_free(ctx_cpu);
    return r;
}

int main(int argc, char ** argv) {
    const bool perf = argc > 1 && strcmp(argv[1], "perf") == 0;
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(cpu, 8);
    ggml_backend_buffer_type_t buft = find_extra_buft("CPU_APPLEAMX");
    if (!buft) { printf("CPU_APPLEAMX buffer type not found\n"); return 1; }

    const ggml_type types[] = { GGML_TYPE_F16, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q4_K, GGML_TYPE_Q8_0 };
    int n_fail = 0, n_tests = 0;
    if (!perf) {
        struct shape { int64_t M, K, N, N2; };
        const shape shapes[] = {
            {32, 256, 1, 1}, {33, 256, 1, 1}, {100, 512, 2, 1}, {64, 256, 3, 1}, {96, 768, 7, 1},
            {65, 256, 8, 1}, {64, 512, 9, 1}, {130, 1024, 33, 1}, {64, 256, 4, 3}, {32, 256, 16, 5},
            {256, 9216, 40, 1}, {200, 17152, 3, 1}, {64, 17152, 64, 1},
        };
        for (ggml_type t : types) {
            for (const auto & s : shapes) {
                result r = run(cpu, buft, t, s.M, s.K, s.N, s.N2, 0);
                const bool ok = r.nmse_amx < 5e-4 && r.nmse_amx < 4*r.nmse_cpu + 1e-6 && std::isfinite(r.nmse_amx);
                n_tests++; n_fail += !ok;
                printf("%-5s M=%-4lld K=%-5lld N=%-3lldx%lld  nmse amx %.2e  cpu %.2e  %s\n", ggml_type_name(t),
                       (long long) s.M, (long long) s.K, (long long) s.N, (long long) s.N2, r.nmse_amx, r.nmse_cpu, ok ? "OK" : "FAIL");
            }
        }
        printf("%d/%d passed\n", n_tests - n_fail, n_tests);
    } else {
        const int64_t Ns[] = { 1, 2, 3, 4, 6, 8, 12, 16, 32 };
        for (ggml_type t : types) {
            for (int64_t N : Ns) {
                result r = run(cpu, buft, t, 2048, 2048*4, N, 1, N >= 128 ? 10 : 100);
                const double gf = 2.0*2048*8192*N*1e-6;
                printf("%-5s M=2048 K=8192 N=%-4lld  amx %8.3f ms (%7.1f GFLOPS)  cpu %8.3f ms (%7.1f GFLOPS)  %.2fx\n", ggml_type_name(t), (long long) N,
                       r.ms_amx, gf/r.ms_amx, r.ms_cpu, gf/r.ms_cpu, r.ms_cpu/r.ms_amx);
            }
        }
    }
    ggml_backend_free(cpu);
    return n_fail ? 1 : 0;
}
