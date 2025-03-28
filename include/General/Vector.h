//
// Created by James Miller on 3/26/2025.
//

#pragma once
#include "../CrossPlatformMacros.h"

namespace MillerPhysics
{
    /// <summary>2D Vector</summary>
    typedef struct MVector2D {
        float x;
        float y;

        MVector2D()
        {
            x = 0.0f;
            y = 0.0f;
        }

        MVector2D(float x, float y)
        {
            this->x = x;
            this->y = y;
        }

        MVector2D(MVector2D const& other)
        {
            x = other.x;
            y = other.y;
        }

        MVector2D& operator+=(const MVector2D& rhs)
        {
            x += rhs.x;
            y += rhs.y;
            return *this;
        }

        MVector2D& operator-=(const MVector2D& rhs)
        {
            x -= rhs.x;
            y -= rhs.y;
            return *this;
        }

        MVector2D& operator*=(const MVector2D& rhs)
        {
            x *= rhs.x;
            y *= rhs.y;
            return *this;
        }

        MVector2D& normalize()
        {
            float magnitude = sqrt(x * x + y * y);
            x /= magnitude;
            y /= magnitude;
            return *this;
        }

        std::string ToString() const
        {
            return std::to_string(x) + ", " + std::to_string(y);
        }

    } MVector2D;

    /// <summary>3D Vector</summary>
    typedef struct MVector
    {
        float x;
        float y;
        float z;

        MVector()
        {
            x = 0;
            y = 0;
            z = 0;
        }

        MVector(float x, float y, float z)
        {
            this->x = x;
            this->y = y;
            this->z = z;
        }

        MVector(const MVector& other)
        {
            x = other.x;
            y = other.y;
            z = other.z;
        }

        explicit MVector(const MVector2D vec2D)
        {
            x = vec2D.x;
            y = vec2D.y;
            z = 0;
        }

        [[nodiscard]] MVector2D ToVector2D() const
        {
            return MVector2D{ x, y };
        }

        MVector operator+=(const MVector& other)
        {
            x += other.x;
            y += other.y;
            z += other.z;
            return *this;
        }

        MVector operator-=(const MVector& other)
        {
            x -= other.x;
            y -= other.y;
            z -= other.z;
            return *this;
        }

        MVector operator*=(const float other)
        {
            x *= other;
            y *= other;
            z *= other;
            return *this;
        }

        MVector& normalize()
        {
            float magnitude = sqrt(x * x + y * y + z * z);
            x /= magnitude;
            y /= magnitude;
            z /= magnitude;
            return *this;
        }

        std::string ToString() const
        {
            return std::to_string(x) + ", " + std::to_string(y) + ", " + std::to_string(z);
        }
    } MVector;

    /// <summary>3D Vector</summary>
    typedef MVector MVector3D;

    /// <summary>4D Vector</summary>
    typedef struct MVector4D {
        float x;
        float y;
        float z;
        float w;

        MVector4D()
        {
            x = 0.0f;
            y = 0.0f;
            z = 0.0f;
            w = 0.0f;
        }

        MVector4D(float x, float y, float z, float w)
        {
            this->x = x;
            this->y = y;
            this->z = z;
            this->w = w;
        }

        explicit MVector4D(const MVector& other)
        {
            this->x = other.x;
            this->y = other.y;
            this->z = other.z;
            this->w = 0.0f;
        }

        explicit MVector4D(const MVector2D& vec2D)
        {
            this->x = vec2D.x;
            this->y = vec2D.y;
            this->z = 0.0f;
            this->w = 0.0f;
        }

        MVector4D(MVector4D const& other)
        {
            this->x = other.x;
            this->y = other.y;
            this->z = other.z;
            this->w = other.w;
        }

        MVector4D& operator=(const MVector& other)
        {
            this->x = other.x;
            this->y = other.y;
            this->z = other.z;
            this->w = 0.0f;
            return *this;
        }

        [[nodiscard]] MVector ToVector3D() const
        {
            return MVector{ x, y, z };
        }

        [[nodiscard]] MVector2D ToVector2D() const
        {
            return MVector2D{ x, y };
        }

        MVector4D& operator+=(const MVector4D& rhs)
        {
            x += rhs.x;
            y += rhs.y;
            z += rhs.z;
            w += rhs.w;
            return *this;
        }

        MVector4D& operator-=(const MVector4D& rhs)
        {
            x -= rhs.x;
            y -= rhs.y;
            z -= rhs.z;
            w -= rhs.w;
            return *this;
        }

        MVector4D& operator*=(float scalar)
        {
            x *= scalar;
            y *= scalar;
            z *= scalar;
            w *= scalar;
            return *this;
        }

        MVector4D& normalize()
        {
            auto magnitude = (float)sqrt(x * x + y * y + z * z);
            x /= magnitude;
            y /= magnitude;
            z /= magnitude;
            w /= magnitude;
            return *this;
        }

        [[nodiscard]] std::string ToString() const
        {
            return std::to_string(x) + ", " + std::to_string(y) + ", " + std::to_string(z) + ", " + std::to_string(w);
        }

    } MVector4D;

    // Vector addition
    /// <summary>Adds the vectors together</summary>
    MVector operator+(const MVector& left, const MVector& right);
    /// <summary>Adds the vectors together</summary>
    MVector2D operator+(const MVector2D& left, const MVector2D& right);
    /// <summary>Adds the vectors together</summary>
    MVector4D operator+(const MVector4D& left, const MVector4D& right);


    // Vector subtraction
    /// <summary>Subtracts the vectors</summary>
    MVector operator-(const MVector& left, const MVector& right);
    /// <summary>Subtracts the vectors</summary>
    MVector2D operator-(const MVector2D& left, const MVector2D& right);
    /// <summary>Subtracts the vectors</summary>
    MVector4D operator-(const MVector4D& left, const MVector4D& right);

    // Vector scalar multiplication
    /// <summary>Multiplies the vectors together</summary>
    MVector operator*(const MVector& left, const float& right);
    /// <summary>Multiplies the vectors together</summary>
    MVector2D operator*(const MVector2D& left, const float& right);
    /// <summary>Multiplies the vectors together</summary>
    MVector4D operator*(const MVector4D& left, const float& right);

    // Vector scalar division
    /// <summary>Divides the vectors</summary>
    MVector operator/(const MVector& left, const float& right);
    /// <summary>Divides the vectors</summary>
    MVector2D operator/(const MVector2D& left, const float& right);
    /// <summary>Divides the vectors</summary>
    MVector4D operator/(const MVector4D& left, const float& right);

    // Vector dot product
    /// <summary>Calculates the dot product of the vectors</summary>
    float operator*(const MVector& left, const MVector& right);
    /// <summary>Calculates the dot product of the vectors</summary>
    float operator*(const MVector2D& left, const MVector2D& right);
    /// <summary>Calculates the dot product of the vectors</summary>
    float operator*(const MVector4D& left, const MVector4D& right);

    // Vector cross product (3D only)
    /// <summary>Calculates the cross product of the vectors</summary>
    MVector operator^(const MVector& left, const MVector& right);

    // Vector normalization
    MVector normalize(MVector vec);
    MVector2D normalize2D(MVector2D vec);
    MVector4D normalize4D(MVector4D vec);

}