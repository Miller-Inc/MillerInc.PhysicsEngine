//
// Created by jwmil on 3/27/2025.
//

#pragma once
#include "../CrossPlatformMacros.h"
#include "../GeneralTypes.h"

namespace MillerPhysics
{
    class MBounds
    {
    public:
        MBounds();
        MBounds(const MVector& min, const MVector& max);
        virtual ~MBounds() = default;

    protected:
        MVector m_min;
        MVector m_max;

    };
};
