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
    this->normalVector = new Vector3(0, 0, 1);
    this->rotation = Quaternion(0, 0, 0, 1);
}

FlatSheet::~FlatSheet() = default;

bool FlatSheet::isColliding(CollisionObject* other)
{
    // TODO: Implement this function
    return false;
}

bool FlatSheet::isTouching(CollisionObject* other)
{
    return false;
}

Vector3 FlatSheet::getClosestPoint(const Vector3& point)
{

    Vector3 norm = getNormalVector();
    float t = Vector3(0.0f, 0.0f, 0.0f).distance(this->position) -
        point.x * norm.x - point.y * norm.y - this->position.z * point.z;

    return {point.x + t * norm.x, point.y + t * norm.y, point.z + t * norm.z};
}

Vector3 FlatSheet::getNormalVector() const
{
    return this->rotation.getNormalVector3();
}
