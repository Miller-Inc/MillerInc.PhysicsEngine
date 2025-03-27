//
// Created by James Miller on 3/27/2025.
//

#pragma once

#include "../CrossPlatformMacros.h"
#include "../GeneralTypes.h"
#include "MBounds.h"

namespace MillerPhysics {

class MObject {
    // Initializers
    public:
    MObject(); // Default constructor

    virtual ~MObject() = default; // Default destructor

    // Fields
    protected:
    // Simple fields
    MVector m_position;
    MQuaternion m_rotation{};
    MVector m_scale;

    // Collision fields
    MBounds m_bounds;



};

} // MillerPhysics

