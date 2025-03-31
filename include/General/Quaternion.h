//
// Created by James Miller on 3/26/2025.
//

#pragma once
#include "Vector.h"

namespace MillerPhysics
{

    typedef struct MQuaternion
    {
        float x, y, z, w;

        MQuaternion& normalize();
        MQuaternion& conjugate();
        MQuaternion& inverse();

        MQuaternion()
        {
            x = y = z = 0;
            w = 1;
        }

        MQuaternion(const float _x, const float _y, const float _z, const float _w)
        {
            x = _x;
            y = _y;
            z = _z;
            w = _w;
        }

        explicit MQuaternion(const MVector4D& vec)
        {
            x = vec.x;
            y = vec.y;
            z = vec.z;
            w = vec.w;
            normalize();
        }

        [[nodiscard]] std::string ToString() const
        {
            return "(" + std::to_string(x) + ", " + std::to_string(y) +
                ", " + std::to_string(z) + ", " + std::to_string(w) + ")";
        }

        [[nodiscard]] float Length() const
        {
            return sqrt(x * x + y * y + z * z + w * w);
        }

        [[nodiscard]] float LengthSquared() const
        {
            return x * x + y * y + z * z + w * w;
        }

        [[nodiscard]] MQuaternion Normalize() const
        {
            const float len = Length();
            if (len == 0)
                return {0, 0, 0, 1};
            return {x / len, y / len, z / len, w / len
        };
        }
    } MQuaternion;

    MQuaternion operator+(const MQuaternion& left, const MQuaternion& right);
    MQuaternion operator-(const MQuaternion& left, const MQuaternion& right);
    MQuaternion operator*(const MQuaternion& left, const float& right);
    MQuaternion operator*(const float& left, const MQuaternion& right);
    MQuaternion operator/(const MQuaternion& left, const float& right);
    MQuaternion operator*(const MQuaternion& left, const MQuaternion& right);


    MQuaternion normalize(const MQuaternion& left);
    MQuaternion conjugate(const MQuaternion& left);
    MQuaternion inverse(const MQuaternion& left);
    MQuaternion lerp(const MQuaternion& start, const MQuaternion& end, const float& percent);
    MQuaternion slerp(const MQuaternion& start, const MQuaternion& end, const float& percent);
    MQuaternion nlerp(const MQuaternion& start, const MQuaternion& end, const float& percent);
    MQuaternion fromEuler(const MVector& euler);
    MVector toEuler(const MQuaternion& quat);
    MQuaternion fromAxisAngle(const MVector& axis, const float& angle);
    std::pair<MVector, float> toAxisAngle(const MQuaternion& quat);

    MQuaternion createRotationQuaternion(const MVector& axis, const float& angle);
    MQuaternion rotateQuaternion(const MQuaternion& q, const MVector& axis, float angle);


} // namespace MillerPhysics