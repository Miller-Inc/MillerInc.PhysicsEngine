//
// Created by James Miller on 3/26/2025.
//

#include "../../include/General/Vector.h"

namespace MillerPhysics
{

    // Vector addition
    MVector operator+(const MVector& left, const MVector& right)
    {
        MVector result;
        result.x = left.x + right.x;
        result.y = left.y + right.y;
        result.z = left.z + right.z;
        return result;
    }

    MVector2D operator+(const MVector2D& left, const MVector2D& right)
    {
        MVector2D result;
        result.x = left.x + right.x;
        result.y = left.y + right.y;
        return result;
    }

    MVector4D operator+(const MVector4D& left, const MVector4D& right)
    {
        MVector4D result;
        result.x = left.x + right.x;
        result.y = left.y + right.y;
        result.z = left.z + right.z;
        result.w = left.w + right.w;
        return result;
    }

    // Vector subtraction
    MVector operator-(const MVector& left, const MVector& right)
    {
        MVector result;
        result.x = left.x - right.x;
        result.y = left.y - right.y;
        result.z = left.z - right.z;
        return result;
    }

    MVector2D operator-(const MVector2D& left, const MVector2D& right)
    {
        MVector2D result;
        result.x = left.x - right.x;
        result.y = left.y - right.y;
        return result;
    }

    MVector4D operator-(const MVector4D& left, const MVector4D& right)
    {
        MVector4D result;
        result.x = left.x - right.x;
        result.y = left.y - right.y;
        result.z = left.z - right.z;
        result.w = left.w - right.w;
        return result;
    }

    // Vector scalar multiplication
    MVector operator*(const MVector& left, const float& right)
    {
        MVector result;
        result.x = left.x * right;
        result.y = left.y * right;
        result.z = left.z * right;
        return result;
    }

    MVector2D operator*(const MVector2D& left, const float& right)
    {
        MVector2D result;
        result.x = left.x * right;
        result.y = left.y * right;
        return result;
    }

    MVector4D operator*(const MVector4D& left, const float& right)
    {
        MVector4D result;
        result.x = left.x * right;
        result.y = left.y * right;
        result.z = left.z * right;
        result.w = left.w * right;
        return result;
    }

    // Vector scalar division
    MVector operator/(const MVector& left, const float& right)
    {
        MVector result;
        result.x = left.x / right;
        result.y = left.y / right;
        result.z = left.z / right;
        return result;
    }

    MVector2D operator/(const MVector2D& left, const float& right)
    {
        MVector2D result;
        result.x = left.x / right;
        result.y = left.y / right;
        return result;
    }

    MVector4D operator/(const MVector4D& left, const float& right)
    {
        MVector4D result;
        result.x = left.x / right;
        result.y = left.y / right;
        result.z = left.z / right;
        result.w = left.w / right;
        return result;
    }

    float operator*(const MVector& left, const MVector& right)
    {
        return left.x * right.x + left.y * right.y + left.z * right.z;
    }

    float operator*(const MVector2D& left, const MVector2D& right)
    {
        return left.x * right.x + left.y * right.y;
    }

    float operator*(const MVector4D& left, const MVector4D& right)
    {
        return left.x * right.x + left.y * right.y + left.z * right.z + left.w * right.w;
    }

    MVector operator^(const MVector& left, const MVector& right)
    {
        return {
            left.y * right.z - left.z * right.y,
            left.z * right.x - left.x * right.z,
            left.x * right.y - left.y * right.x
        };
    }

    // Vector normalization
    MVector normalize(MVector vec)
    {
        float magnitude = std::sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
        return { vec.x / magnitude, vec.y / magnitude, vec.z / magnitude };
    }

    MVector2D normalize2D(MVector2D vec)
    {
        float magnitude = std::sqrt(vec.x * vec.x + vec.y * vec.y);
        return { vec.x / magnitude, vec.y / magnitude };
    }

    MVector4D normalize4D(MVector4D vec)
    {
        float magnitude = std::sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z + vec.w * vec.w);
        return { vec.x / magnitude, vec.y / magnitude, vec.z / magnitude, vec.w / magnitude };
    }



}