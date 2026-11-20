#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int positive_int(const char* arg, int maximum) {
    std::size_t used = 0;
    long long value = std::stoll(arg, &used);
    if (used != std::string(arg).size() || value < 1 || value > maximum)
        throw std::invalid_argument("integer outside supported range");
    return static_cast<int>(value);
}

int main(int argc, char** argv) {
    try {
        if (argc > 4) throw std::invalid_argument("usage: vector_bench [N [rounds [repeats]]]");
        const int n = argc > 1 ? positive_int(argv[1], 1 << 26) : 1 << 20;
        const int rounds = argc > 2 ? positive_int(argv[2], 1000) : 5;
        const int repeats = argc > 3 ? positive_int(argv[3], 100000) : 100;
        const int warmup = 100;
        const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
        std::vector<float> a(n), b(n), c(n);
        for (int i = 0; i < n; ++i) {
            a[i] = static_cast<float>(i % 17 - 8);
            b[i] = static_cast<float>((i % 23) * 3 - 11);
        }
        float *da, *db, *dc;
        CUDA_CHECK(cudaMalloc(&da, bytes));
        CUDA_CHECK(cudaMalloc(&db, bytes));
        CUDA_CHECK(cudaMalloc(&dc, bytes));
        CUDA_CHECK(cudaMemcpy(da, a.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(db, b.data(), bytes, cudaMemcpyHostToDevice));
        const int threads = 256, blocks = (n + threads - 1) / threads;
        vectorAdd<<<blocks, threads>>>(da, db, dc, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(c.data(), dc, bytes, cudaMemcpyDeviceToHost));
        bool ok = true;
        for (int i = 0; i < n; ++i) {
            // Small integer-valued floats: this sum is exactly representable.
            if (c[i] != a[i] + b[i]) {
                std::fprintf(stderr, "FAIL N=%d i=%d got=%g expected=%g\n", n, i, c[i], a[i] + b[i]);
                ok = false;
                break;
            }
        }
        if (ok) {
            std::fprintf(stderr, "Correctness N=%d PASS (all elements)\n", n);
            for (int i = 0; i < warmup; ++i) vectorAdd<<<blocks, threads>>>(da, db, dc, n);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            cudaEvent_t start, end;
            CUDA_CHECK(cudaEventCreate(&start));
            CUDA_CHECK(cudaEventCreate(&end));
            std::puts("n,round,repeats,warmup,block_size,avg_us,effective_gbps");
            for (int r = 1; r <= rounds; ++r) {
                CUDA_CHECK(cudaEventRecord(start));
                for (int i = 0; i < repeats; ++i) vectorAdd<<<blocks, threads>>>(da, db, dc, n);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaEventRecord(end));
                CUDA_CHECK(cudaEventSynchronize(end));
                float ms;
                CUDA_CHECK(cudaEventElapsedTime(&ms, start, end));
                const double us = ms * 1000.0 / repeats;
                const double gbps = (3.0 * bytes / 1e9) / (us / 1e6);
                std::printf("%d,%d,%d,%d,%d,%.6f,%.6f\n", n, r, repeats, warmup, threads, us, gbps);
            }
            CUDA_CHECK(cudaEventDestroy(start));
            CUDA_CHECK(cudaEventDestroy(end));
        }
        CUDA_CHECK(cudaFree(da));
        CUDA_CHECK(cudaFree(db));
        CUDA_CHECK(cudaFree(dc));
        return ok ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return EXIT_FAILURE;
    }
}

