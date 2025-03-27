//
// Created by James Miller on 3/26/2025.
//

#include "../../include/General/Quaternion.h"
#include <cmath>

namespace MillerPhysics
{
    // Quaternion addition
    MQuaternion operator+(const MQuaternion& left, const MQuaternion& right)
    {
        MQuaternion result;
        result.x = left.x + right.x;
        result.y = left.y + right.y;
        result.z = left.z + right.z;
        result.w = left.w + right.w;
        return result;
    }

    // Quaternion subtraction
    MQuaternion operator-(const MQuaternion& left, const MQuaternion& right)
    {
        MQuaternion result;
        result.x = left.x - right.x;
        result.y = left.y - right.y;
        result.z = left.z - right.z;
        result.w = left.w - right.w;
        return result;
    }

    // Quaternion scalar multiplication
    MQuaternion operator*(const MQuaternion& left, const float& right)
    {
        MQuaternion result;
        result.x = left.x * right;
        result.y = left.y * right;
        result.z = left.z * right;
        result.w = left.w * right;
        return result;
    }

    MQuaternion operator*(const float& left, const MQuaternion& right)
    {
        MQuaternion result;
        result.x = left * right.x;
        result.y = left * right.y;
        result.z = left * right.z;
        result.w = left * right.w;
        return result;
    }

    // Quaternion scalar division
    MQuaternion operator/(const MQuaternion& left, const float& right)
    {
        MQuaternion result;
        result.x = left.x / right;
        result.y = left.y / right;
        result.z = left.z / right;
        result.w = left.w / right;
        return result;
    }

    // Quaternion dot product
    MQuaternion operator*(const MQuaternion& left, const MQuaternion& right)
    {
        MQuaternion result;
        result.w = left.w * right.w - left.x * right.x - left.y * right.y - left.z * right.z;
        result.x = left.w * right.x + left.x * right.w + left.y * right.z - left.z * right.y;
        result.y = left.w * right.y - left.x * right.z + left.y * right.w + left.z * right.x;
        result.z = left.w * right.z + left.x * right.y - left.y * right.x + left.z * right.w;
        return result;
    }

    MQuaternion normalize(const MQuaternion& q)
    {
        float magnitude = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
        return { q.x / magnitude, q.y / magnitude, q.z / magnitude, q.w / magnitude };
    }

    MQuaternion& MQuaternion::normalize()
    {
        float magnitude = std::sqrt(x * x + y * y + z * z + w * w);
        x /= magnitude;
        y /= magnitude;
        z /= magnitude;
        w /= magnitude;
        return *this;
    }

    MQuaternion conjugate(const MQuaternion& q)
    {
        return { -q.x, -q.y, -q.z, q.w };
    }

    MQuaternion& MQuaternion::conjugate()
    {
        this->x = -this->x;
        this->y = -this->y;
        this->z = -this->z;
        return *this;
    }

    MQuaternion inverse(const MQuaternion& q)
    {
        float normSquared = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
        if (normSquared > 0.0f)
        {
            float invNorm = 1.0f / normSquared;
            return { -q.x * invNorm, -q.y * invNorm, -q.z * invNorm, q.w * invNorm };
        }
        return q; // Return the original quaternion if norm is zero
    }

    MQuaternion& MQuaternion::inverse()
    {
        float normSquared = x * x + y * y + z * z + w * w;
        if (normSquared > 0.0f)
        {
            float invNorm = 1.0f / normSquared;
            this->x = -this->x * invNorm;
            this->y = -this->y * invNorm;
            this->z = -this->z * invNorm;
            this->w = this->w * invNorm;
        }
        return *this;
    }

    MQuaternion lerp(const MQuaternion& start, const MQuaternion& end, const float& percent)
    {
        MQuaternion result;
        result.x = start.x + percent * (end.x - start.x);
        result.y = start.y + percent * (end.y - start.y);
        result.z = start.z + percent * (end.z - start.z);
        result.w = start.w + percent * (end.w - start.w);
        return normalize(result);
    }

    MQuaternion slerp(const MQuaternion& start, const MQuaternion& end, const float& percent)
    {
        // Compute the cosine of the angle between the two quaternions
        float dot = start.x * end.x + start.y * end.y + start.z * end.z + start.w * end.w;

        // If the dot product is negative, slerp won't take the shorter path.
        // Fix by reversing one quaternion.
        MQuaternion endCopy = end;
        if (dot < 0.0f)
        {
            dot = -dot;
            endCopy.x = -endCopy.x;
            endCopy.y = -endCopy.y;
            endCopy.z = -endCopy.z;
            endCopy.w = -endCopy.w;
        }

        constexpr float DOT_THRESHOLD = 0.9995f;
        if (dot > DOT_THRESHOLD)
        {
            // If the quaternions are very close, use linear interpolation
            MQuaternion result = start + percent * (endCopy - start);
            return normalize(result);
        }

        // Calculate the angle between the quaternions
        float theta_0 = std::acos(dot);
        float theta = theta_0 * percent;

        // Compute the second quaternion
        MQuaternion q2 = endCopy - start * dot;
        q2 = normalize(q2);

        // Compute the interpolated quaternion
        MQuaternion result = start * std::cos(theta) + q2 * std::sin(theta);
        return result;
    }

    MVector toEuler(const MQuaternion& quat)
    {
        MVector euler;

        // Roll (x-axis rotation)
        float sinr_cosp = 2 * (quat.w * quat.x + quat.y * quat.z);
        float cosr_cosp = 1 - 2 * (quat.x * quat.x + quat.y * quat.y);
        euler.x = std::atan2(sinr_cosp, cosr_cosp);

        // Pitch (y-axis rotation)
        float sinp = 2 * (quat.w * quat.y - quat.z * quat.x);
        if (std::abs(sinp) >= 1)
            euler.y = std::copysign( M_PI / 2, sinp); // Use 90 degrees if out of range
        else
            euler.y = std::asin(sinp);

        // Yaw (z-axis rotation)
        float siny_cosp = 2 * (quat.w * quat.z + quat.x * quat.y);
        float cosy_cosp = 1 - 2 * (quat.y * quat.y + quat.z * quat.z);
        euler.z = std::atan2(siny_cosp, cosy_cosp);

        return euler;
    }

    MQuaternion fromEuler(const MVector& euler)
    {
        float cy = std::cos(euler.z * 0.5f);
        float sy = std::sin(euler.z * 0.5f);
        float cp = std::cos(euler.y * 0.5f);
        float sp = std::sin(euler.y * 0.5f);
        float cr = std::cos(euler.x * 0.5f);
        float sr = std::sin(euler.x * 0.5f);

        MQuaternion q;
        q.w = cr * cp * cy + sr * sp * sy;
        q.x = sr * cp * cy - cr * sp * sy;
        q.y = cr * sp * cy + sr * cp * sy;
        q.z = cr * cp * sy - sr * sp * cy;

        return q;
    }

    MQuaternion fromAxisAngle(const MVector& axis, const float& angle)
    {
        MQuaternion q;
        float halfAngle = angle * 0.5f;
        float s = std::sin(halfAngle);
        q.w = std::cos(halfAngle);
        q.x = axis.x * s;
        q.y = axis.y * s;
        q.z = axis.z * s;
        return q;
    }

    // Convert quaternion to axis-angle
    std::pair<MVector, float> toAxisAngle(const MQuaternion& quat)
    {
        MVector axis;
        float angle = 2.0f * std::acos(quat.w);
        float s = std::sqrt(1.0f - quat.w * quat.w); // assuming quaternion is normalized

        if (s < 0.001f) // to avoid division by zero
        {
            axis.x = quat.x;
            axis.y = quat.y;
            axis.z = quat.z;
        }
        else
        {
            axis.x = quat.x / s;
            axis.y = quat.y / s;
            axis.z = quat.z / s;
        }

        return std::make_pair(axis, angle);
    }


    MQuaternion createRotationQuaternion(const MVector& axis, const float& angle)
    {
        MVector normalizedAxis = normalize(axis);
        float halfAngle = angle / 2.0f;
        float sinHalfAngle = std::sin(halfAngle);

        MQuaternion rotation;
        rotation.w = std::cos(halfAngle);
        rotation.x = normalizedAxis.x * sinHalfAngle;
        rotation.y = normalizedAxis.y * sinHalfAngle;
        rotation.z = normalizedAxis.z * sinHalfAngle;

        return rotation;
    }

    MQuaternion rotateQuaternion(const MQuaternion& quat, const MVector& axis, const float& angle)
    {
        MQuaternion rotation = createRotationQuaternion(axis, angle);
        MQuaternion rotationConjugate = { -rotation.x, -rotation.y, -rotation.z, rotation.w };

        MQuaternion result = rotation * quat * rotationConjugate;
        return result;
    }



}