#include <iostream>
#include <cstdio>   // fprintf
#include <cstdlib> // malloc、free、exit
#include <cuda_runtime.h>
using namespace std;

// 实验一：证明 vector add 在边界上正确
//N	blocks=(N+255)/256	启动线程总数	不写结果的线程数
// 1	  1	              256	        255
// 255	1	              256	        1
// 256	1	              256	        0
// 257	2	              512	        255
// 1000	4	              1024	      24
// 1025	5	              1280	      255


#define CUDA_CHECK(call) do {                           \
    cudaError_t err = (call);                           \
    if (err != cudaSuccess) {                           \
        fprintf(stderr, "CUDA error: %s\n",             \
                cudaGetErrorString(err));                \
        exit(EXIT_FAILURE);                             \
    }                                                    \
} while (0)

__global__ void vector_add(int *a, int *b, int *c, int n)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n){
      c[i]=a[i]+b[i];
    }
}

bool run_case(int n){
  int *a, *b, *c;
  a=(int*)malloc(n*sizeof(int));
  b=(int*)malloc(n*sizeof(int));
  c=(int*)malloc(n*sizeof(int));

  if (!a || !b || !c) {
    std::cerr << "Host allocation failed\n";
    free(a);
    free(b);
    free(c);
    return false;
}

  for(int i=0;i<n;i++){
    a[i] = i % 17 - 8;
    b[i] = (i * 3) % 23 - 11;
  }

  int *d_a, *d_b, *d_c;
  CUDA_CHECK(cudaMalloc((void**)&d_a,n*sizeof(int)));
  CUDA_CHECK(cudaMalloc((void**)&d_b,n*sizeof(int)));
  CUDA_CHECK(cudaMalloc((void**)&d_c,n*sizeof(int)));

  CUDA_CHECK(cudaMemcpy(d_a,a,n*sizeof(int),cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b,b,n*sizeof(int),cudaMemcpyHostToDevice));


  int blockSize=256;
  int numBlocks=(n+blockSize-1)/blockSize;
  vector_add<<<numBlocks,blockSize>>>(d_a,d_b,d_c,n);

  //第一步检查启动错误；同步使异步执行错误在这里被发现。
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(c,d_c,n*sizeof(int),cudaMemcpyDeviceToHost));
  
  bool ok = true;
  for (int i = 0; i < n; ++i) {
      int expected = a[i] + b[i];
      if (c[i] != expected) {
          std::cerr << "N=" << n << " i=" << i
                    << " got=" << c[i] << " expected=" << expected << '\n';
          ok = false;
          break;
      }
  }

  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));

  free(a);
  free(b);
  free(c);
  
  return ok;
}

int main() {
    const int sizes[] = {1, 255, 256, 257, 1000, 1025};
    bool all_ok = true;

    for (int n : sizes) {
        bool ok = run_case(n);
        cout << "N=" << n << " " << (ok ? "PASS" : "FAIL") << '\n';

        if (!ok) {
            all_ok = false;
        }
    }

    return all_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}