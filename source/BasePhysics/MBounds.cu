//
// Created by James Miller on 3/29/2025.
//

#if CUDA_AVAILABLE
#include "../../include/BasePhysics/MBounds.h"

namespace MillerPhysics
{
    __global__ void intersectsKernel(const MVector* gpu_min, const MVector* gpu_max, const MVector* bounds_min, const MVector* bounds_max, bool* result)
    {
        *result = (gpu_min->x <= bounds_max->x && gpu_max->x >= bounds_min->x &&
                   gpu_min->y <= bounds_max->y && gpu_max->y >= bounds_min->y &&
                   gpu_min->z <= bounds_max->z && gpu_max->z >= bounds_min->z);
    }

    bool MBounds::intersects(const MBounds& bounds) const
    {
        bool result;
        bool* gpu_result;
        cudaMalloc(&gpu_result, sizeof(bool));

        intersectsKernel<<<128, 128>>>(gpu_min, gpu_max, bounds.gpu_min, bounds.gpu_max, gpu_result);

        cudaMemcpy(&result, gpu_result, sizeof(bool), cudaMemcpyDeviceToHost);
        cudaFree(gpu_result);

        return result;
    }

    MBounds::MBounds()
    {
        m_max = {0.0f, 0.0f, 0.0f};
        m_min = {0.0f, 0.0f, 0.0f};
        m_center = {0.0f, 0.0f, 0.0f};

        gpu_center = nullptr;
        gpu_min = nullptr;
        gpu_max = nullptr;

        cudaMalloc(&gpu_center, sizeof(MVector));
        cudaMalloc(&gpu_min, sizeof(MVector));
        cudaMalloc(&gpu_max, sizeof(MVector));
        cudaMemcpy(gpu_center, &m_center, sizeof(MVector), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_min, &m_min, sizeof(MVector), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_max, &m_max, sizeof(MVector), cudaMemcpyHostToDevice);
    }

    MBounds::MBounds(const MVector& min, const MVector& max)
    {
        gpu_center = nullptr;
        gpu_min = nullptr;
        gpu_max = nullptr;
        m_max = max;
        m_min = min;
        m_center = {(min.x + max.x) / 2.0f, (min.y + max.y) / 2.0f, (min.z + max.z) / 2.0f};
        cudaMemcpy(gpu_center, &m_center, sizeof(MVector), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_min, &m_min, sizeof(MVector), cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_max, &m_max, sizeof(MVector), cudaMemcpyHostToDevice);
    }

    MVector MBounds::GetCenter()
    {
        cudaMemcpy(&m_center, gpu_center, sizeof(MVector), cudaMemcpyDeviceToHost);
        return m_center;
    }

    MVector MBounds::GetMin()
    {
        cudaMemcpy(&m_min, gpu_min, sizeof(MVector), cudaMemcpyDeviceToHost);
        return m_min;
    }

    MVector MBounds::GetMax()
    {
        cudaMemcpy(&m_max, gpu_max, sizeof(MVector), cudaMemcpyDeviceToHost);
        return m_max;
    }

    void MBounds::SetCenter(const MVector& center)
    {
        cudaMemcpy(gpu_center, &m_center, sizeof(MVector), cudaMemcpyHostToDevice);
        m_center = center;
    }

    void MBounds::SetMin(const MVector& min)
    {
        cudaMemcpy(gpu_min, &m_min, sizeof(MVector), cudaMemcpyHostToDevice);
        m_min = min;
    }

    void MBounds::SetMax(const MVector& max)
    {
        cudaMemcpy(gpu_max, &m_max, sizeof(MVector), cudaMemcpyHostToDevice);
        m_max = max;
    }



}

#endif