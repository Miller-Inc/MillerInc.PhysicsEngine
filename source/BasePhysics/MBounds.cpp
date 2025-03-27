//
// Created by James Miller on 3/27/2025.
//

#include "../../include/BasePhysics/MBounds.h"

namespace MillerPhysics
{
    MBounds::MBounds()
    {
        m_max = MVector(0, 0, 0);
        m_min = MVector(0, 0, 0);
    }

    MBounds::MBounds(const MVector& min, const MVector& max)
    {
        m_max = max;
        m_min = min;
    }
};