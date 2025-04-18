//
// Created by James Miller on 3/27/2025.
//

#include "../../include/BasePhysics/MBounds.h"

namespace MillerPhysics
{
    #if !CUDA_AVAILABLE
    MBounds::MBounds()
    {
        m_center = {0, 0, 0};
        AddBoundPoint(m_center);
    }

    bool MBounds::intersects(const MBounds& bounds) const
    {
        // TODO: Implement intersection logic for CPU
        return false; // Implement intersection logic here
    }

    MVector MBounds::GetCenter()
    {
        return m_center;
    }

    void MBounds::SetCenter(const MVector& center)
    {
        m_center = center;
    }

    #endif

    void MBounds::AddBoundPoint(const MVector& point)
    {
        if (m_num_points == 0)
        {
            m_points = (MVector*)malloc(sizeof(MVector));
        }
        else
        {
            m_points = (MVector*)realloc(m_points, sizeof(MVector) * (m_num_points + 1));
        }

        memcpy((m_points + m_num_points), &point, sizeof(MVector));
        m_num_points++;
        m_has_changed = true;

        #if CUDA_AVAILABLE
        CopyToGPU(); // Copy the points to the GPU
        #endif
    }

    MBounds MBounds::DefaultMBounds(const MVector& center)
    {
        MBounds bounds;
        bounds.m_center = center;
        bounds.AddBoundPoint(center); // Add the center point
        return bounds;
    }

    MBounds MBounds::DefaultCubeBounds(const MVector& center, float size)
    {
        MBounds bounds;
        bounds.m_center = center;

        // Creates the 8 points of the cube
        const MVector lower_left_back(center.x - size / 2, center.y - size / 2, center.z - size / 2);
        const MVector upper_right_front(center.x + size / 2, center.y + size / 2, center.z + size / 2);
        const MVector lower_left_front(center.x - size / 2, center.y - size / 2, center.z + size / 2);
        const MVector upper_right_back(center.x + size / 2, center.y + size / 2, center.z - size / 2);
        const MVector lower_right_back(center.x + size / 2, center.y - size / 2, center.z - size / 2);
        const MVector lower_right_front(center.x + size / 2, center.y - size / 2, center.z + size / 2);
        const MVector upper_left_back(center.x - size / 2, center.y + size / 2, center.z - size / 2);
        const MVector upper_left_front(center.x - size / 2, center.y + size / 2, center.z + size / 2);

        // Add the points to the bounds list
        bounds.AddBoundPoint(lower_left_back);
        bounds.AddBoundPoint(upper_right_front);
        bounds.AddBoundPoint(lower_left_front);
        bounds.AddBoundPoint(upper_right_back);
        bounds.AddBoundPoint(lower_right_back);
        bounds.AddBoundPoint(lower_right_front);
        bounds.AddBoundPoint(upper_left_back);
        bounds.AddBoundPoint(upper_left_front);

        return bounds;
    }

    MBounds MBounds::DefaultSphereBounds(const MVector& center, float radius, uint16_t num_points)
    {
        MBounds bounds;
        bounds.m_center = center;

        if (num_points == 0)
        {
            // Default to a high definition sphere if no number of points is provided
            num_points = 128; // 128 points for a high-resolution sphere
        }

        // Create points on the sphere surface
        for (uint16_t i = 0; i < num_points; ++i)
        {
            float theta = static_cast<float>(i) * (M_PI * 2.0f / num_points);
            float phi = static_cast<float>(i) * (M_PI / num_points);
            float x = radius * sin(phi) * cos(theta);
            float y = radius * sin(phi) * sin(theta);
            float z = radius * cos(phi);
            bounds.AddBoundPoint(MVector(x, y, z));
        }

        return bounds;
    }

    MBounds MBounds::DefaultCapsuleBounds(const MVector& center, float radius, float height, uint16_t num_points)
    {
        MBounds bounds;
        bounds.m_center = center;
        bounds.m_num_points = num_points;

        // Allocate memory for points
        bounds.m_points = (MVector*)malloc(num_points * sizeof(MVector));

        uint16_t num_cylinder_points = num_points / 2;
        uint16_t num_hemisphere_points = num_points / 4;

        float half_height = height / 2.0f;

        // Generate cylinder points
        for (uint16_t i = 0; i < num_cylinder_points; ++i)
        {
            float angle = 2.0f * M_PI * (i / (float)num_cylinder_points);
            bounds.m_points[i] = {
                center.x + radius * cos(angle),
                center.y + radius * sin(angle),
                center.z + (i % 2 == 0 ? half_height : -half_height)
            };
        }

        // Generate top hemisphere points
        for (uint16_t i = 0; i < num_hemisphere_points; ++i)
        {
            float phi = M_PI * (i / (float)num_hemisphere_points) / 2.0f;
            float theta = 2.0f * M_PI * (i / (float)num_hemisphere_points);
            bounds.m_points[num_cylinder_points + i] = {
                center.x + radius * sin(phi) * cos(theta),
                center.y + radius * sin(phi) * sin(theta),
                center.z + half_height + radius * cos(phi)
            };
        }

        // Generate bottom hemisphere points
        for (uint16_t i = 0; i < num_hemisphere_points; ++i)
        {
            float phi = M_PI * (i / (float)num_hemisphere_points) / 2.0f;
            float theta = 2.0f * M_PI * (i / (float)num_hemisphere_points);
            bounds.m_points[num_cylinder_points + num_hemisphere_points + i] = {
                center.x + radius * sin(phi) * cos(theta),
                center.y + radius * sin(phi) * sin(theta),
                center.z - half_height - radius * cos(phi)
            };
        }

        return bounds;
    }

    MBounds MBounds::DefaultCylinderBounds(const MVector& center)
    {
        MBounds bounds;
        bounds.m_center = center;

        // Create points on the cylinder surface
        for (uint16_t i = 0; i < 8; ++i)
        {
            float angle = static_cast<float>(i) * (M_PI * 2.0f / 8);
            float x = cos(angle);
            float y = sin(angle);
            bounds.AddBoundPoint(MVector(x, y, 0));
            bounds.AddBoundPoint(MVector(x, y, 1)); // Add top point
        }

        return bounds;
    }
};