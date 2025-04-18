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

        for (size_t i = 0; i < m_objects.size(); i++)
        {
            if (m_objects[i] == nullptr)
            {
                std::cerr << "Null pointer encountered at index: " << i << std::endl;
                continue;
            }
            else
            {
                for (size_t j = i + 1; j < m_objects.size(); j++)
                {
                    if (m_objects[j] == nullptr)
                    {
                        std::cerr << "Null pointer encountered at index: " << j << std::endl;
                        continue;
                    }
                    else
                    {
                        if (m_objects[i]->m_bounds->intersects(*m_objects[j]->m_bounds))
                        {
                            std::cout << "Collision detected between objects " << i << " and " << j << std::endl;
                        }
                    }
                }
            }
        }

        /*
        int blockSize = 256;
        int numBlocks = (m_objects.size() + blockSize - 1) / blockSize;
        if (numBlocks <= 0)
        {
            numBlocks = 1;
        }

        // Allocate device memory
        MObject** d_objects;
        CUDA_CHECK_ERROR(cudaMalloc(&d_objects, m_objects.size() * sizeof(MObject*)));
        bool* gpu_result;
        CUDA_CHECK_ERROR(cudaMalloc(&gpu_result, m_objects.size() * sizeof(bool)));

        // Copy data to device
        CUDA_CHECK_ERROR(cudaMemcpy(d_objects, m_objects.data(), m_objects.size() * sizeof(MObject*), cudaMemcpyHostToDevice));

        // Launch kernel
        checkCollisionsKernel<<<numBlocks, blockSize>>>(d_objects, (uint8_t)m_objects.size(), gpu_result);
        CUDA_CHECK_ERROR(cudaGetLastError()); // Check for kernel launch errors
        CUDA_CHECK_ERROR(cudaDeviceSynchronize()); // Wait for the kernel to finish

        // Free device memory
        CUDA_CHECK_ERROR(cudaFree(gpu_result));
        CUDA_CHECK_ERROR(cudaFree(d_objects));
        std::cout << "End MScene::checkCollisions" << std::endl; */
    }
}

#endif