//
// Created by James Miller on 10/9/2024.
//

#ifndef CUBE_H
#define CUBE_H

#pragma once

#include "../CollisionObject.h"


class Cube : public CollisionObject
{

public:
    // Constructors
    Cube();
    explicit Cube(float sideLength);

    // Destructors
    ~Cube() override;

    // Variables
    float sideLength;

    // Override Functions
    bool isColliding(CollisionObject* other) override;

    bool isTouching(CollisionObject* other) override;

    Vector3 getClosestPoint(const Vector3& point) override;


};

#endif //CUBE_H
