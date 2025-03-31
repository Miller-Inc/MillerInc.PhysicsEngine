//
// Created by James Miller on 3/27/2025.
//

#include "../../include/BasePhysics/MBounds.h"

namespace MillerPhysics
{
    #if !CUDA_AVAILABLE
    MBounds::MBounds()
    {
        m_max = MVector(0, 0, 0);
        m_min = MVector(0, 0, 0);
        m_center = MVector(0, 0, 0);
    }

    MBounds::MBounds(const MVector& min, const MVector& max)
    {
        m_max = max;
        m_min = min;
        m_center = (m_max + m_min) / 2;
    }
    bool MBounds::intersects(const MBounds& bounds) const
    {
        return (m_min.x <= bounds.m_max.x && m_max.x >= bounds.m_min.x) &&
               (m_min.y <= bounds.m_max.y && m_max.y >= bounds.m_min.y) &&
               (m_min.z <= bounds.m_max.z && m_max.z >= bounds.m_min.z);
    }

    MVector MBounds::GetCenter()
    {
        return m_center;
    }

    MVector MBounds::GetMin()
    {
        return m_min;
    }

    MVector MBounds::GetMax()
    {
        return m_max;
    }

    void MBounds::SetCenter(const MVector& center)
    {
        m_center = center;
    }

    void MBounds::SetMin(const MVector& min)
    {
        m_min = min;
    }

    void MBounds::SetMax(const MVector& max)
    {
        m_max = max;
    }

    #endif
};