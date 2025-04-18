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

        [[nodiscard]] std::string ToString() const
        {
            return std::to_string(x) + ", " + std::to_string(y);
        }

        [[nodiscard]] float Length() const
        {
            return (float)sqrt(x * x + y * y);
        }

        [[nodiscard]] float LengthSquared() const
        {
            return x * x + y * y;
        }

        [[nodiscard]] float Dot(const MVector2D& other) const
        {
            return x * other.x + y * other.y;
        }

        [[nodiscard]] MVector2D Cross(const MVector2D& other) const
        {
            return {
                y * other.x - x * other.y,
                x * other.y - y * other.x};
        }

        [[nodiscard]] float Angle(const MVector2D& other) const
        {
            return acos(Dot(other) / (Length() * other.Length()));
        }

        [[nodiscard]] float AngleDegrees(const MVector2D& other) const
        {
            return Angle(other) * (180.0f / M_PI);
        }

        [[nodiscard]] float AngleRadians(const MVector2D& other) const
        {
            return Angle(other);
        }

        [[nodiscard]] MVector2D Perpendicular() const
        {
            return { -y, x };
        }

        [[nodiscard]] MVector2D Normalized() const
        {
            float magnitude = Length();
            return { x / magnitude, y / magnitude };
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

        explicit MVector(const MVector2D& vec2D)
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

        [[nodiscard]] std::string ToString() const
        {
            return std::to_string(x) + ", " + std::to_string(y) + ", " + std::to_string(z);
        }

        [[nodiscard]] float Length() const
        {
            return (float)sqrt(x * x + y * y + z * z);
        }

        [[nodiscard]] float LengthSquared() const
        {
            return x * x + y * y + z * z;
        }

        [[nodiscard]] float Dot(const MVector& other) const
        {
            return x * other.x + y * other.y + z * other.z;
        }

        [[nodiscard]] MVector Cross(const MVector& other) const
        {
            return {
                y * other.z - z * other.y,
                z * other.x - x * other.z,
                x * other.y - y * other.x};
        }

        __host__ __device__ bool operator==(const MVector& other) const
        {
            return (fabs(x - other.x) < 0.0001f &&
                    fabs(y - other.y) < 0.0001f &&
                    fabs(z - other.z) < 0.0001f);
        }

        __host__ __device__ bool operator!=(const MVector& other) const
        {
            return !(*this == other);
        }

        __host__ __device__ MVector operator+(const MVector& other) const
        {
            return {x + other.x, y + other.y, z + other.z};
        }

        __host__ __device__ MVector operator-(const MVector& other) const
        {
            return {x - other.x, y - other.y, z - other.z};
        }

        __host__ __device__ MVector operator*(float other) const
        {
            return {x * other, y * other, z * other};
        }

        __host__ __device__ MVector operator/(float other) const
        {
            return {x / other, y / other, z / other};
        }

        __host__ __device__ bool operator<(const MVector& other) const
        {
            return (x < other.x && y < other.y && z < other.z);
        }
        __host__ __device__ bool operator>(const MVector& other) const
        {
            return (x > other.x && y > other.y && z > other.z);
        }
        __host__ __device__ bool operator<=(const MVector& other) const
        {
            return (x <= other.x && y <= other.y && z <= other.z);
        }
        __host__ __device__ bool operator>=(const MVector& other) const
        {
            return (x >= other.x && y >= other.y && z >= other.z);
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

        [[nodiscard]] float Length() const
        {
            return (float)sqrt(x * x + y * y + z * z + w * w);
        }

        [[nodiscard]] float LengthSquared() const
        {
            return x * x + y * y + z * z + w * w;
        }

        [[nodiscard]] float Dot(const MVector4D& other) const
        {
            return x * other.x + y * other.y + z * other.z + w * other.w;
        }

        [[nodiscard]] MVector4D Cross(const MVector4D& other) const
        {
            return {
                y * other.z - z * other.y,
                z * other.x - x * other.z,
                x * other.y - y * other.x,
                0.0f};
        }

    } MVector4D;

    // Vector addition
    /// <summary>Adds the vectors together</summary>
    MVector2D operator+(const MVector2D& left, const MVector2D& right);
    /// <summary>Adds the vectors together</summary>
    MVector4D operator+(const MVector4D& left, const MVector4D& right);


    // Vector subtraction
    /// <summary>Subtracts the vectors</summary>
    MVector2D operator-(const MVector2D& left, const MVector2D& right);
    /// <summary>Subtracts the vectors</summary>
    MVector4D operator-(const MVector4D& left, const MVector4D& right);

    // Vector scalar multiplication
    /// <summary>Multiplies the vectors together</summary>
    MVector operator*(const float& left, const MVector& right);
    /// <summary>Multiplies the vectors together</summary>
    MVector2D operator*(const MVector2D& left, const float& right);
    MVector2D operator*(const float& left, const MVector2D& right);
    /// <summary>Multiplies the vectors together</summary>
    MVector4D operator*(const MVector4D& left, const float& right);
    MVector4D operator*(const float& left, const MVector4D& right);

    // Vector scalar division
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

    bool operator==(const MVector2D& left, const MVector2D& right);
    bool operator==(const MVector4D& left, const MVector4D& right);
    bool operator!=(const MVector2D& left, const MVector2D& right);
    bool operator!=(const MVector4D& left, const MVector4D& right);

    bool operator<(const MVector2D& left, const MVector2D& right);
    bool operator<(const MVector4D& left, const MVector4D& right);
    bool operator>(const MVector2D& left, const MVector2D& right);
    bool operator>(const MVector4D& left, const MVector4D& right);
    bool operator<=(const MVector2D& left, const MVector2D& right);
    bool operator<=(const MVector4D& left, const MVector4D& right);
    bool operator>=(const MVector2D& left, const MVector2D& right);
    bool operator>=(const MVector4D& left, const MVector4D& right);

    // Vector normalization
    MVector normalize(MVector vec);
    MVector2D normalize2D(MVector2D vec);
    MVector4D normalize4D(MVector4D vec);

}