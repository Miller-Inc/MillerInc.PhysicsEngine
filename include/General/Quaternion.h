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