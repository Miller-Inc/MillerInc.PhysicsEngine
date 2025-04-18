//
// Created by jwmil on 3/27/2025.
//

#pragma once
#include <vector>

#include "../CrossPlatformMacros.h"
#include "../GeneralTypes.h"

namespace MillerPhysics
{
    class MBounds
    {
    public:
        MBounds();
        [[nodiscard]] bool intersects(const MBounds& bounds) const;

        std::vector<MVector> GetBoundPoints();
        MVector GetCenter();
        void SetCenter(const MVector& center);
        void AddBoundPoint(const MVector& point);
        void UpdateBounds(const MVector& new_center, MQuaternion new_relative_orientation);

        /// <summary>
        ///  Returns a default MBounds object with the given center. (a single point)
        /// </summary>
        static MBounds DefaultMBounds(const MVector& center);

        /// <summary>
        ///  Returns a default MBounds that relates to a cube
        /// </summary>
        static MBounds DefaultCubeBounds(const MVector& center, float size = 1.0f);

        /// <summary>
        ///  Returns a default MBounds that relates to a sphere
        /// </summary>
        static MBounds DefaultSphereBounds(const MVector& center, float radius = 1.0f, uint16_t num_points = 0);

        /// <summary>
        ///  Returns a default MBounds that relates to a capsule
        /// </summary>
        static MBounds DefaultCapsuleBounds(const MVector& center, float radius = 1.0f, float height = 1.0f, uint16_t num_points = 0);

        /// <summary>
        ///  Returns a default MBounds that relates to a cylinder
        /// </summary>
        static MBounds DefaultCylinderBounds(const MVector& center);

        #if CUDA_AVAILABLE
        MVector* gpu_points{};
        MVector* gpu_center{};
        virtual ~MBounds();
        #else
        virtual ~MBounds() = default;
        #endif

    protected:

        #if CUDA_AVAILABLE
        /// <summary>
        ///  Used to copy the MBounds to the GPU
        /// </summary>
        /// <remarks>
        ///  Call before using the MBounds in a CUDA kernel
        /// </remarks>
        virtual void CopyToGPU();

        virtual void UpdatePointsFromGPU();
        #endif

        /// <summary>
        ///  The points that define the bounds of the object
        /// </summary>
        /// <remarks>
        /// Calculate the bounds of the object based on the points
        /// </remarks>
        MVector* m_points{};

        /// <summary>
        ///  Used to find how to calculate the bounds of the object and what the inside is defined as
        /// </summary>
        /// <remarks>
        ///  This is mainly used when calculating the bounds of the object,
        ///     and is copied over to the GPU when using CUDA
        /// </remarks>
        MVector m_center{};
        uint16_t m_num_points_min = 0;
        uint16_t m_num_points_max = 0;
        uint16_t m_num_points = 0;

        bool m_has_changed = true; // Used to determine if the bounds have changed since the last time they were calculated
        bool m_gpu_changed = false;

        friend class MScene;

    };

};
