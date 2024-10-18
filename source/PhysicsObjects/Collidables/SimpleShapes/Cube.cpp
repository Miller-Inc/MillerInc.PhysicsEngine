//
// Created by James Miller on 10/9/2024.
//

#include "../../../../include/PhysicsObjects/Collidables/SimpleShapes/Cube.h"

Cube::Cube() : Cube::Cube(1.0f)
{
    // Default Constructor
}

Cube::Cube(float sideLength) : CollisionObject()
{
    // Constructor
}

Cube::~Cube() = default;

bool Cube::isColliding(CollisionObject* other)
{

}

bool Cube::isTouching(CollisionObject* other)
{

}

Vector3 Cube::getClosestPoint(const Vector3& point)
{

}

