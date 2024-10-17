//
// Created by James Miller on 10/8/2024.
//

#include "../../../include/FieldTypes/BaseTypes/Vector3.h"

#include <cmath>
#include <complex>

// Constructors Section
Vector3::Vector3(float x, float y, float z)
{
    this->x = x;
    this->y = y;
    this->z = z;
}

// Default constructor
Vector3::Vector3() : Vector3(0, 0, 0)
{

}

// Operators Section

Vector3 Vector3::operator+(const Vector3& other) const
{
    return {x + other.x, y + other.y, z + other.z};
}

Vector3 Vector3::operator-(const Vector3& other) const
{
    return {x - other.x, y - other.y, z - other.z};
}

Vector3 Vector3::operator*(float scalar) const
{
    return {x * scalar, y * scalar, z * scalar};
}

Vector3 Vector3::operator/(float scalar) const
{
    return {x / scalar, y / scalar, z / scalar};
}

Vector3& Vector3::operator+=(const Vector3& other)
{
    x += other.x;
    y += other.y;
    z += other.z;
    return *this;
}

Vector3& Vector3::operator-=(const Vector3& other)
{
    x -= other.x;
    y -= other.y;
    z -= other.z;
    return *this;
}

Vector3& Vector3::operator*=(float scalar)
{
    x *= scalar;
    y *= scalar;
    z *= scalar;
    return *this;
}

bool Vector3::operator==(const Vector3& other) const
{
    return std::fabs(x - other.x) < 0.0000001f && std::fabs(y - other.y) < 0.0000001f && std::fabs(z - other.z) < 0.0000001f;
}

Vector3 Vector3::cross(const Vector3& other) const
{
    return {y * other.z - z * other.y, z * other.x - x * other.z, x * other.y - y * other.x};
}

float Vector3::dot(const Vector3& other) const
{
    return x * other.x + y * other.y + z * other.z;
}

Vector3 Vector3::cross(const Vector3& a, const Vector3& b)
{
    return a.cross(b);
}

float Vector3::dot(const Vector3& a, const Vector3& b)
{
    return a.dot(b);
}

/// <summary>
/// Returns the readable string representation of the vector
/// </summary>
std::string Vector3::toString() const
{
    return "(" + std::to_string(x) + ", " + std::to_string(y) + ", " + std::to_string(z) + ")";
}

float Vector3::distance(const Vector3& other) const
{
    return (*this - other).magnitude();
}

float Vector3::magnitude() const
{
    return std::sqrt(x * x + y * y + z * z);
}
