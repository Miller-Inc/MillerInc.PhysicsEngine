//
// Created by James Miller on 10/9/2024.
//

#include "../../../../include/PhysicsObjects/Collidables/SimpleShapes/FlatSheet.h"

FlatSheet::FlatSheet() : FlatSheet::FlatSheet(1.0f, 1.0f)
{
    // Default Constructor
}

FlatSheet::FlatSheet(float sideLength) : FlatSheet::FlatSheet(sideLength, sideLength)
{
    // Constructor
}

FlatSheet::FlatSheet(float height, float width)
: FlatSheet(height, width, Vector3(0, 0, 0), 1.0f)
{
    // Constructor
}

FlatSheet::FlatSheet(float height, float width, Vector3 position, float mass) : CollisionObject()
{
    // Constructor
    this->height = height;
    this->width = width;
    this->position = position;
    this->mass = mass;
}

FlatSheet::~FlatSheet() = default;

bool FlatSheet::isColliding(CollisionObject* other)
{
    return false;
}

bool FlatSheet::isTouching(CollisionObject* other)
{
    return false;
}

Vector3 FlatSheet::getClosestPoint(const Vector3& point)
{
    return position;
}

Vector3 FlatSheet::getNormalVector() const
{
    return this->rotation.getNormalVector3();
}
