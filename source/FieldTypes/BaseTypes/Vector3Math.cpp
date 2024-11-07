//
// Created by James  on 11Miller/6/2024.
//

#include "../../../include/FieldTypes/BaseTypes/Vector3Math.h"
#include <stdexcept>

#if CUDA_AVAILABLE



#else

void Vector3Math::addVectors(const std::vector<Vector3>& a, const std::vector<Vector3>& b, std::vector<Vector3>& result) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must be of the same length");
    }

    unsigned int n = a.size();
    result.resize(n);

    for (unsigned int i = 0; i < n; ++i) {
        result[i].x = a[i].x + b[i].x;
        result[i].y = a[i].y + b[i].y;
        result[i].z = a[i].z + b[i].z;
    }
}

void Vector3Math::multiplyVectors(const std::vector<Vector3>& a, const std::vector<float>& b, std::vector<Vector3>& result) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors and scalars must be of the same length");
    }

    unsigned int n = a.size();
    result.resize(n);

    for (unsigned int i = 0; i < n; ++i) {
        result[i].x = a[i].x * b[i];
        result[i].y = a[i].y * b[i];
        result[i].z = a[i].z * b[i];
    }
}

void Vector3Math::multiplyVectors(const std::vector<Vector3>& a, const std::vector<Vector3>& b, std::vector<Vector3>& result) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must be of the same length");
    }

    unsigned int n = a.size();
    result.resize(n);

    for (unsigned int i = 0; i < n; ++i) {
        result[i].x = a[i].x * b[i].x;
        result[i].y = a[i].y * b[i].y;
        result[i].z = a[i].z * b[i].z;
    }
}

#endif