//
// Created by James Miller on 10/11/2024.
//

#include "../../../include/FieldTypes/BaseTypes/Quaternion.h"
#include <cmath>
#include <bits/algorithmfwd.h>


Quaternion* Quaternion::getNormalVector() const
{
    float magnitude = std::sqrt(x * x + y * y + z * z + w * w);
    if (magnitude == 0) return nullptr; // Handle zero magnitude case
    return new Quaternion(x / magnitude, y / magnitude, z / magnitude, w / magnitude);
}

Quaternion* Quaternion::getConjugate() const
{
    return new Quaternion(-x, -y, -z, w);
}

// Normalization
Quaternion Quaternion::normalize() const {
    float magnitude = std::sqrt(x * x + y * y + z * z + w * w);
    if (magnitude == 0) return Quaternion(0, 0, 0, 1); // Return identity quaternion if magnitude is zero
    return Quaternion(x / magnitude, y / magnitude, z / magnitude, w / magnitude);
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
    Quaternion qv(v.x, v.y, v.z, 0);
    Quaternion result = *this * qv * inverse();
    return Vector3(result.x, result.y, result.z);
}

// From Axis-Angle
Quaternion Quaternion::fromAxisAngle(const Vector3& axis, float angle) {
    float halfAngle = angle / 2.0f;
    float s = std::sin(halfAngle);
    return Quaternion(axis.x * s, axis.y * s, axis.z * s, std::cos(halfAngle));
}

// To Axis-Angle
void Quaternion::toAxisAngle(Vector3& axis, float& angle) const {
    if (w > 1) normalize(); // Normalize if w is greater than 1
    angle = 2 * std::acos(w);
    float s = std::sqrt(1 - w * w); // Assuming quaternion is normalized
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
