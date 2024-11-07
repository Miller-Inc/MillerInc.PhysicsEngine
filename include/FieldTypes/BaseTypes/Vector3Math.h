//
// Created by James Miller on 11/6/2024.
//

#ifndef VECTOR3MATH_H
#define VECTOR3MATH_H

#include <vector>
#include "Vector3.h"

class Vector3Math {
public:
    static void addVectors(const std::vector<Vector3>& a, const std::vector<Vector3>& b, std::vector<Vector3>& result);
    static void multiplyVectors(const std::vector<Vector3>& a, const std::vector<float>& b, std::vector<Vector3>& result);
    static void multiplyVectors(const std::vector<Vector3>& a, const std::vector<Vector3>& b, std::vector<Vector3>& result);
};

#endif // VECTOR3MATH_H
