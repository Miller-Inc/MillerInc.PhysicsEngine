//
// Created by James Miller on 3/29/2025.
//

#if CUDA_AVAILABLE
#include "../../include/BasePhysics/MBounds.h"

namespace MillerPhysics
{
    /// <summary>
    /// Checks all the bounding points to see if they intersect
    /// </summary>
    __global__ void intersectsKernel(MVector* obj1_bounds, MVector* obj2_bounds,
        const MVector* obj1Center, bool* result, const uint8_t num_vertices1,
        const uint8_t num_vertices2)
    {
        const uint8_t i = blockIdx.x;
        const uint8_t j = threadIdx.x;
        const uint8_t k = blockIdx.x * blockDim.x + threadIdx.x;

        // printf("i = %d, j = %d, k = %d\n", i, j, k);

        if (i < num_vertices1 && j < num_vertices2 && k)
        {
            // Needs redone to check if the bounding point is between the other bounding point and the center
            result[k] = (obj2_bounds[j] <= obj1_bounds[i]);

            // Check if the point is between the center and the other bounding point
            // Check x coordinate
            if (obj1_bounds[i].x <= obj2_bounds[j].x && obj2_bounds[j].x <= obj1Center->x)
            {
                if (obj1_bounds[i].y <= obj2_bounds[j].y && obj2_bounds[j].y <= obj1Center->y)
                {
                    if (obj1_bounds[i].z <= obj2_bounds[j].z && obj2_bounds[j].z <= obj1Center->z)
                    {
                        result[k] = true;
                    } else if (obj1_bounds[i].z >= obj2_bounds[j].z && obj2_bounds[j].z >= obj1Center->z)
                    {
                        result[k] = true;
                    } else
                    {
                        result[k] = false;
                    }
                }
                else if (obj1_bounds[i].y >= obj2_bounds[j].y && obj2_bounds[j].y >= obj1Center->y)
                {
                    if (obj1_bounds[i].z <= obj2_bounds[j].z && obj2_bounds[j].z <= obj1Center->z)
                    {
                        result[k] = true;
                    } else if (obj1_bounds[i].z >= obj2_bounds[j].z && obj2_bounds[j].z >= obj1Center->z)
                    {
                        result[k] = true;
                    } else
                    {
                        result[k] = false;
                    }
                }
            } else if (obj1_bounds[i].x >= obj2_bounds[j].x && obj2_bounds[j].x >= obj1Center->x)
            {
                if (obj1_bounds[i].y <= obj2_bounds[j].y && obj2_bounds[j].y <= obj1Center->y)
                {
                    if (obj1_bounds[i].z <= obj2_bounds[j].z && obj2_bounds[j].z <= obj1Center->z)
                    {
                        result[k] = true;
                    } else if (obj1_bounds[i].z >= obj2_bounds[j].z && obj2_bounds[j].z >= obj1Center->z)
                    {
                        result[k] = true;
                    } else
                    {
                        result[k] = false;
                    }
                }
                else if (obj1_bounds[i].y >= obj2_bounds[j].y && obj2_bounds[j].y >= obj1Center->y)
                {
                    if (obj1_bounds[i].z <= obj2_bounds[j].z && obj2_bounds[j].z <= obj1Center->z)
                    {
                        result[k] = true;
                    } else if (obj1_bounds[i].z >= obj2_bounds[j].z && obj2_bounds[j].z >= obj1Center->z)
                    {
                        result[k] = true;
                    } else
                    {
                        result[k] = false;
                    }
                }
            }

            if (result[k])
            {
                printf("Collision detected between objects %d and %d\n", i, j);
            }
        }
    }

    ///<summary>
    /// Checks all the results to make sure that they are all true, proving intersection
    ///</summary>
    __global__ void checkIntersectionsKernel(const bool* results, bool * const found,  bool* final_result)
    {
        const uint8_t i = blockIdx.x * blockDim.x + threadIdx.x;

        // Check if there is already something that proves no intersection
        if (*found)
        {
            return;
        }

        printf("Results[%d] = %d\n", i, results[i]);

        // Check if the result is true for this index
        if (results[i]) {
            *final_result = true; // Ensure the final result is true
            *found = true; // Set found to true
        } else {
            *final_result = false;
            *found = true;
        }

    }

    void MBounds::CopyToGPU()
    {
        if (!m_has_changed)
        {
            return;
        }

        cudaFree(gpu_points); // Free the previous GPU memory if it exists

        // Allocate GPU memory
        if (cudaMalloc(&gpu_points, sizeof(MVector) * m_num_points) ||
            cudaMalloc(&gpu_center, sizeof(MVector)) != cudaSuccess) {
            std::cerr << "CUDA malloc failed in CopyToGPU" << std::endl;
            exit(EXIT_FAILURE);
            }

        // Copy data to GPU
        cudaMemcpy(gpu_points, m_points, sizeof(MVector) * m_num_points, cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_center, &m_center, sizeof(MVector), cudaMemcpyHostToDevice);

        m_has_changed = false;
        m_gpu_changed = false; // Reset the GPU changed flag
    }

    void MBounds::UpdatePointsFromGPU()
    {
        if (!m_gpu_changed)
        {
            return;
        }
        // Copy data from GPU to CPU cache
        cudaMemcpy(m_points, gpu_points, sizeof(MVector) * m_num_points, cudaMemcpyDeviceToHost);
        cudaMemcpy(&m_center, gpu_center, sizeof(MVector), cudaMemcpyDeviceToHost);
        m_has_changed = false;
    }


    bool MBounds::intersects(const MBounds& bounds) const
    {
        if (bounds.m_num_points == 0)
        {
            return false;
        }

        size_t num_checks = sizeof(bool) * m_num_points * bounds.m_num_points;
        bool result = false;
        bool* gpu_result;
        bool* gpu_final_result;
        bool found = false;
        bool* found_ptr; // Pointer to the found variable on the GPU

        cudaMalloc(&found_ptr, sizeof(bool));
        cudaMalloc(&gpu_result, num_checks);
        cudaMalloc(&gpu_final_result, sizeof(bool));
        cudaMemcpy(gpu_final_result, &result, sizeof(bool), cudaMemcpyHostToDevice);
        cudaMemcpy(found_ptr, &found, sizeof(bool), cudaMemcpyHostToDevice);

        intersectsKernel<<<m_num_points, bounds.m_num_points>>>(gpu_points, bounds.gpu_points,
            (gpu_center), gpu_result, m_num_points_max,
            bounds.m_num_points_max);

        // Check for kernel launch errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "Kernel launch error: " << cudaGetErrorString(err) << std::endl;
        }

        cudaDeviceSynchronize();

        checkIntersectionsKernel<<<num_checks, 1>>>(gpu_result, found_ptr, gpu_final_result);

        cudaDeviceSynchronize();

        // cudaMemcpy(&result, gpu_result, num_checks, cudaMemcpyDeviceToHost);
        cudaFree(gpu_result);
        cudaMemcpy(&result, gpu_final_result, sizeof(bool), cudaMemcpyDeviceToHost);
        cudaFree(gpu_final_result);

        fflush(stdout);
        return result;
    }

    MBounds::MBounds()
    {
        m_center = {0.0f, 0.0f, 0.0f};
        AddBoundPoint({0, 0, 0});

        MBounds::CopyToGPU();
    }

    MBounds::~MBounds()
    {
        // Free GPU memory
        cudaFree(gpu_center);
        cudaFree(gpu_points);
        free(gpu_points);
    }

    MVector MBounds::GetCenter()
    {
        cudaMemcpy(&m_center, gpu_center, sizeof(MVector), cudaMemcpyDeviceToHost);
        return m_center;
    }

    void MBounds::SetCenter(const MVector& center)
    {
        m_center = center;
        cudaMemcpy(gpu_center, &m_center, sizeof(MVector), cudaMemcpyHostToDevice);
    }

    std::vector<MVector> MBounds::GetBoundPoints()
    {
        std::vector<MVector> points(m_num_points);
        if (gpu_points == nullptr)
        {
            return points;
        }

        MBounds::UpdatePointsFromGPU();

        return points;
    }

    void MBounds::UpdateBounds(const MVector& new_center, const MQuaternion new_relative_orientation)
    {
        // Recalculate the bounds based on the new center and orientation
        MVector change = new_center - m_center;
        for (size_t i = 0; i < m_num_points; i++)
        {
            m_points[i] += change;
            m_points[i] = new_relative_orientation * m_points[i];
        }

        m_center = new_center;
    }


}

#endif