#include "../../../include/FieldTypes/BaseTypes/Vector3.h"
#include <cmath>

// Constructors Section
Vector3::Vector3(float x, float y, float z) : x(x), y(y), z(z) {}
Vector3::Vector3() : Vector3(0, 0, 0) {}

std::byte* Vector3::toBytes() const
{
    std::byte bytes[96];
    bytes[0] = static_cast<std::byte>(x);
    bytes[32] = static_cast<std::byte>(y);
    bytes[64] = static_cast<std::byte>(z);
    return bytes;
}

// Pointer-based Operators Section
Vector3* Vector3::operator+(const Vector3* other) const {
    return new Vector3(x + other->x, y + other->y, z + other->z);
}

Vector3* Vector3::operator-(const Vector3* other) const {
    return new Vector3(x - other->x, y - other->y, z - other->z);
}

Vector3* Vector3::operator*(const float* scalar) const {
    return new Vector3(x * *scalar, y * *scalar, z * *scalar);
}

Vector3* Vector3::operator/(const float* scalar) const {
    return new Vector3(x / *scalar, y / *scalar, z / *scalar);
}

Vector3* Vector3::operator*=(const float* scalar) {
    x *= *scalar;
    y *= *scalar;
    z *= *scalar;
    return this;
}

Vector3* Vector3::operator+=(const Vector3* other) {
    x += other->x;
    y += other->y;
    z += other->z;
    return this;
}

Vector3* Vector3::operator-=(const Vector3* other) {
    x -= other->x;
    y -= other->y;
    z -= other->z;
    return this;
}
bool Vector3::operator==(const Vector3* other) const {
    return std::fabs(x - other->x) < 0.0000001f && std::fabs(y - other->y) < 0.0000001f && std::fabs(z - other->z) < 0.0000001f;
}

// Reference-based Operators Section
Vector3 Vector3::operator+(const Vector3& other) const {
    return {x + other.x, y + other.y, z + other.z};
}

Vector3 Vector3::operator-(const Vector3& other) const {
    return {x - other.x, y - other.y, z - other.z};
}

Vector3 Vector3::operator*(float scalar) const {
    return {x * scalar, y * scalar, z * scalar};
}

Vector3 Vector3::operator/(float scalar) const {
    return {x / scalar, y / scalar, z / scalar};
}

Vector3& Vector3::operator+=(const Vector3& other) {
    x += other.x;
    y += other.y;
    z += other.z;
    return *this;
}

Vector3& Vector3::operator-=(const Vector3& other) {
    x -= other.x;
    y -= other.y;
    z -= other.z;
    return *this;
}

Vector3& Vector3::operator*=(float scalar) {
    x *= scalar;
    y *= scalar;
    z *= scalar;
    return *this;
}

bool Vector3::operator==(const Vector3& other) const {
    return std::fabs(x - other.x) < 0.0000001f && std::fabs(y - other.y) < 0.0000001f && std::fabs(z - other.z) < 0.0000001f;
}

// Other Methods
Vector3 Vector3::cross(const Vector3* other) const {
    return {y * other->z - z * other->y, z * other->x - x * other->z, x * other->y - y * other->x};
}

Vector3 Vector3::cross(const Vector3& other) const {
    return {y * other.z - z * other.y, z * other.x - x * other.z, x * other.y - y * other.x};
}

float Vector3::dot(const Vector3* other) const {
    return x * other->x + y * other->y + z * other->z;
}

float Vector3::dot(const Vector3& other) const {
    return x * other.x + y * other.y + z * other.z;
}

Vector3 Vector3::cross(const Vector3* a, const Vector3* b) {
    return a->cross(b);
}

Vector3 Vector3::cross(const Vector3& a, const Vector3& b) {
    return a.cross(b);
}

float Vector3::dot(const Vector3* a, const Vector3* b) {
    return a->dot(b);
}

float Vector3::dot(const Vector3& a, const Vector3& b) {
    return a.dot(b);
}

std::string Vector3::toString() const {
    return "(" + std::to_string(x) + ", " + std::to_string(y) + ", " + std::to_string(z) + ")";
}

float Vector3::distance(const Vector3* other) const {
    Vector3 diff = *this - *other;
    return diff.magnitude();
}

float Vector3::distance(const Vector3& other) const {
    Vector3 diff = *this - other;
    return diff.magnitude();
}

float Vector3::length() const
{
    return magnitude();
}

Vector3 Vector3::normalize()
{
    const float len = magnitude();
    if (len > 0) {
        x /= len;
        y /= len;
        z /= len;
    }
    return *this;
}

float Vector3::magnitude() const {
    return std::sqrt(x * x + y * y + z * z);
}
