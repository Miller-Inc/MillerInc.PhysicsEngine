//
// Created by James Miller on 10/11/2024.
//

#include "../../../include/FieldTypes/BaseTypes/Quaternion.h"
#include <cmath>
#include <algorithm>


Quaternion Quaternion::getNormalVector() const
{
    float magnitude = std::sqrt(x * x + y * y + z * z + w * w);
    if (magnitude == 0) return {0, 0, 0, 0}; // Handle zero magnitude case
    return {x / magnitude, y / magnitude, z / magnitude, w / magnitude};
}

Quaternion* Quaternion::getNormalVectorP() const
{
    float magnitude = std::sqrt(x * x + y * y + z * z + w * w);
    if (magnitude == 0) return new Quaternion(0, 0, 0, 0); // Handle zero magnitude case
    return new Quaternion(x / magnitude, y / magnitude, z / magnitude, w / magnitude);
}

// Normalization
[[nodiscard]] Quaternion* Quaternion::normalizeP() {
    const float magnitude = std::sqrt(x * x + y * y + z * z + w * w);
    if (magnitude == 0)
    {
        x = 0;
        y = 0;
        z = 0;
        w = 1;
    }// Return identity quaternion if magnitude is zero
    else
    {
        x /= magnitude;
        y /= magnitude;
        z /= magnitude;
        w /= magnitude;
    }
    return this;
}

// Slerp (Spherical Linear Interpolation)
Quaternion Quaternion::slerp(const Quaternion& other, float t) const {
    float dotProduct = dot(other);
    dotProduct = std::clamp(dotProduct, -1.0f, 1.0f); // Clamp dot product to avoid acos domain errors

    float theta = std::acos(dotProduct) * t;
    Quaternion relativeQuat = (other - *this * dotProduct).normalize();
    return *this * std::cos(theta) + relativeQuat * std::sin(theta);
}

// Rotation
Vector3 Quaternion::rotate(const Vector3& v) const {
    const Quaternion qv(v.x, v.y, v.z, 0);
    Quaternion result = *this * qv * inverse();
    return {result.x, result.y, result.z};
}

// From Axis-Angle
Quaternion Quaternion::fromAxisAngle(const Vector3& axis, float angle) {
    float halfAngle = angle / 2.0f;
    float s = std::sin(halfAngle);
    return {axis.x * s, axis.y * s, axis.z * s, std::cos(halfAngle)};
}

// To Axis-Angle
void Quaternion::toAxisAngle(Vector3& axis, float& angle) {
    Quaternion o = this->normalize();

    if (w > 1) o = *this->normalizeP(); // Normalize if w is greater than 1
    angle = 2 * std::acos(w);
    float s = std::sqrt(1 - o.w * o.w); // Assuming quaternion is normalized
    if (s < 0.001) { // To avoid division by zero
        axis = Vector3(x, y, z);
    } else {
        axis = Vector3(x / s, y / s, z / s);
    }
}

Vector3 Quaternion::getNormalVector3() const
{
    float magnitude = std::sqrt(x * x + y * y + z * z + w * w);
    if (magnitude == 0) return {0, 0, 0}; // Handle zero magnitude case
    return {x / magnitude, y / magnitude, z / magnitude};
}

std::string Quaternion::toString() const
{
    return "(" + std::to_string(x) + ", " + std::to_string(y) + ", " + std::to_string(z) + ", " + std::to_string(w) + ")";
}

std::byte* Quaternion::toBytes() const
{
    auto* bytes = new std::byte[sizeof(float) * 4];
    std::memcpy(bytes, &x, sizeof(float));
    std::memcpy(bytes + sizeof(float), &y, sizeof(float));
    std::memcpy(bytes + 2 * sizeof(float), &z, sizeof(float));
    std::memcpy(bytes + 3 * sizeof(float), &w, sizeof(float));
    return bytes;
}

Quaternion Quaternion::getConjugate() const
{
    return {-x, -y, -z, w};
}

Vector3* Quaternion::rotate(const Vector3* v) const
{
    return new Vector3(rotate(*v));
}

Quaternion* Quaternion::slerp(const Quaternion* other, float t) const
{
    return new Quaternion(slerp(*other, t));
}


