#include <cuda_runtime.h>
#include "gpu_kernels.h"

__global__ void updatePositions(float* positions, float* velocities, int numObjects, float timestep) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numObjects) {
        positions[idx * 3] += velocities[idx * 3] * timestep;
        positions[idx * 3 + 1] += velocities[idx * 3 + 1] * timestep;
        positions[idx * 3 + 2] += velocities[idx * 3 + 2] * timestep;
    }
}

void launchUpdatePositions(float* positions, float* velocities, int numObjects, float timestep) {
    int blockSize = 256;
    int numBlocks = (numObjects + blockSize - 1) / blockSize;
    updatePositions<<<numBlocks, blockSize>>>(positions, velocities, numObjects, timestep);
    cudaDeviceSynchronize();
}
