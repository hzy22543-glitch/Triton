#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call) do {                           \
    cudaError_t err = (call);                           \
    if (err != cudaSuccess) {                           \
        fprintf(stderr, "CUDA error: %s\n",             \
                cudaGetErrorString(err));                \
        exit(EXIT_FAILURE);                             \
    }                                                    \
} while (0)

__global__ void vectorAdd(
    const float* A,
    const float* B,
    float* C,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

void benchmark(int N) {
    size_t bytes = (size_t)N * sizeof(float);

    //内存分配
    float *A, *B, *C;
    CUDA_CHECK(cudaMalloc(&A, bytes));
    CUDA_CHECK(cudaMalloc(&B, bytes));
    CUDA_CHECK(cudaMalloc(&C, bytes));

    //初始化
    CUDA_CHECK(cudaMemset(A, 0, bytes));
    CUDA_CHECK(cudaMemset(B, 0, bytes));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    // Warmup预热
    for (int i = 0; i < 20; ++i) {
        vectorAdd<<<blocks, threads>>>(A, B, C, N);
    }
    //同步
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));

    const int repeats = 100;

    //start event
    CUDA_CHECK(cudaEventRecord(start));

    //重复启动
    for (int i = 0; i < repeats; ++i) {
        vectorAdd<<<blocks, threads>>>(A, B, C, N);
    }

    //end event
    CUDA_CHECK(cudaEventRecord(end));
    //等待同步
    CUDA_CHECK(cudaEventSynchronize(end));
    CUDA_CHECK(cudaGetLastError());
    
    float total_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(
        &total_ms, start, end
    ));
    //计算平均时间
    float avg_ms = total_ms / repeats;

    // 假设每个元素读取 A/B，写入 C
    //乘以 3 是因为 Vector Add 对每个元素需要进行 3 次数据访问：读取 A、读取 B、写入 C。
    //除以 \(10^9\) 是把 Byte 转换为 GB
    double gb = 3.0 * bytes / 1e9;
    double bandwidth = gb / (avg_ms / 1000.0);

    printf("N=%d | avg=%.4f ms | effective BW=%.2f GB/s\n",
           N, avg_ms, bandwidth);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(end));
    CUDA_CHECK(cudaFree(A));
    CUDA_CHECK(cudaFree(B));
    CUDA_CHECK(cudaFree(C));
}


int main(int argc, char** argv) {
  if (argc > 1) {
        //atoi:将字符串转换为整数
        int N = atoi(argv[1]);
        if (N <= 0) {
            fprintf(stderr, "N must be positive\n");
            return 1;
        }
        benchmark(N);
    } else {
        benchmark(1 << 10);
        benchmark(1 << 18);
        benchmark(1 << 19);
        benchmark(1 << 20);
        benchmark(1 << 21);
        benchmark(1 << 22);
        benchmark(1 << 23);
        benchmark(1 << 24);
    }
    return 0;
}//意思是2的24次方

    


