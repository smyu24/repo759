#include <cstdio>
#include <cuda_runtime.h>
#include <cstdlib>
#include <ctime>
#include <random>


__global__ void kernel(int *dA, int a) {
    int x = threadIdx.x;
    int y = blockIdx.x;
    int idx = y * blockDim.x + x;
    dA[idx] = a * x + y;
}

int main() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(1, 10); 
    int a = dis(gen);

    int *dA;
    cudaMalloc(&dA, 16 * sizeof(int));
    kernel<<<2, 8>>>(dA, a);
    
    int hA[16];
    cudaMemcpy(hA, dA, 16 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dA);
    
    for (int i = 0; i < 16; ++i) {
        std::printf("%d ", hA[i]);
    }
    std::printf("\n");
    
    return 0;
}
