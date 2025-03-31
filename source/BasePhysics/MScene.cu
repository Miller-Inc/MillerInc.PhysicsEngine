//
// Created by James Miller on 3/29/2025.
//

#if CUDA_AVAILABLE

#include "../../include/BasePhysics/MScene.h"

namespace MillerPhysics
{
    __global__ void checkCollisionsKernel(MObject** objects, uint8_t numObjects, bool* collide) {
        const uint8_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < numObjects) {
            for (int j = i + 1; j < numObjects; ++j) {
                if (objects[i] == nullptr || objects[j] == nullptr) {
                    printf("Null pointer encountered at i: %d, j: %d\n", i, j);
                    continue;
                }
                collide[i] = (objects[i]->m_bounds->gpu_min->x <= objects[j]->m_bounds->gpu_max->x &&
                              objects[i]->m_bounds->gpu_max->x >= objects[j]->m_bounds->gpu_min->x &&
                              objects[i]->m_bounds->gpu_min->y <= objects[j]->m_bounds->gpu_max->y &&
                              objects[i]->m_bounds->gpu_max->y >= objects[j]->m_bounds->gpu_min->y &&
                              objects[i]->m_bounds->gpu_min->z <= objects[j]->m_bounds->gpu_max->z &&
                              objects[i]->m_bounds->gpu_max->z >= objects[j]->m_bounds->gpu_min->z);
                if (collide[i]) {
                    printf("Collision detected between objects %d and %d\n", i, j);
                }
            }
        }
    }

#include <iostream>
#include <cuda_runtime.h>

#define CUDA_CHECK_ERROR(call) \
do { \
cudaError_t err = call; \
if (err != cudaSuccess) { \
std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " \
<< cudaGetErrorString(err) << std::endl; \
exit(EXIT_FAILURE); \
} \
} while (0)

    void MScene::checkCollisions() const
    {
        std::cout << "MScene::checkCollisions" << std::endl;
        int blockSize = 256;
        int numBlocks = (m_objects.size() + blockSize - 1) / blockSize;
        if (numBlocks <= 0)
        {
            numBlocks = 1;
        }
        MObject** d_objects;
        CUDA_CHECK_ERROR(cudaMalloc(&d_objects, m_objects.size() * sizeof(MObject*)));
        bool* gpu_result;
        CUDA_CHECK_ERROR(cudaMalloc(&gpu_result, m_objects.size() * sizeof(bool) * blockSize * numBlocks));
        CUDA_CHECK_ERROR(cudaMemcpy(d_objects, m_objects.data(), m_objects.size() * sizeof(MObject*), cudaMemcpyHostToDevice));

        checkCollisionsKernel<<<numBlocks, blockSize>>>(d_objects, (uint8_t)m_objects.size(), gpu_result);
        CUDA_CHECK_ERROR(cudaGetLastError()); // Check for kernel launch errors
        CUDA_CHECK_ERROR(cudaDeviceSynchronize()); // Wait for the kernel to finish

        CUDA_CHECK_ERROR(cudaFree(gpu_result));
        CUDA_CHECK_ERROR(cudaFree(d_objects));
        std::cout << "End MScene::checkCollisions" << std::endl;
    }
}

#endif