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
        [[nodiscard]] bool intersects(const MBounds& bounds) const;

        MVector GetMin();
        MVector GetMax();
        MVector GetCenter();
        void SetMin(const MVector& min);
        void SetMax(const MVector& max);
        void SetCenter(const MVector& center);

        #if CUDA_AVAILABLE
        MVector* gpu_min;
        MVector* gpu_max;
        MVector* gpu_center;
        #endif

    protected:
        MVector m_min;
        MVector m_max;
        MVector m_center;

        friend class MScene;

    };
};
