//
// Created by James Miller on 3/27/2025.
//

#include "../../include/BasePhysics/MObject.h"

namespace MillerPhysics {
    MObject::MObject()
    {
        m_position = MVector(0, 0, 0);
        m_rotation = MQuaternion(0, 0, 0, 1);
        m_scale = MVector(1, 1, 1);
        m_bounds = MBounds(m_position - (m_scale / 2), m_position + (m_scale / 2));
    }
} // MillerPhysics