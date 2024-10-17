//
// Created by James Miller on 10/9/2024.
//

#include "../../../../include/PhysicsObjects/Collidables/SimpleShapes/FlatSheet.h"

#include <cmath>

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

    // Set the corners of the sheet
    this->topRight = Vector3(width / 2, height / 2, 0);
    this->topLeft = Vector3(-width / 2, height / 2, 0);
    this->bottomRight = Vector3(width / 2, -height / 2, 0);
    this->bottomLeft = Vector3(-width / 2, -height / 2, 0);

    // Set the rotation of the sheet
    this->normalVector = new Vector3(0, 0, 1);
    this->rotation = Quaternion(0, 0, 0, 1);
}

FlatSheet::~FlatSheet() = default;

/// <summary>
/// Checks to see if the sheet is colliding with the other object
/// </summary>
bool FlatSheet::isColliding(CollisionObject* other)
{
    Vector3 thisClosestPoint = this->getClosestPoint(other->position);
    Vector3 otherClosestPoint = other->getClosestPoint(thisClosestPoint);
    if (std::fabs(other->position.x) > std::fabs(thisClosestPoint.x) &&
        std::fabs(other->position.y) > std::fabs(thisClosestPoint.y) &&
        std::fabs(other->position.z) > std::fabs(thisClosestPoint.z) &&
        std::fabs(otherClosestPoint.x) < std::fabs(this->position.x) &&
        std::fabs(otherClosestPoint.y) < std::fabs(this->position.y) &&
        std::fabs(otherClosestPoint.z) < std::fabs(this->position.z))
    {
        return true;
    }
    return false;
}

/// <summary>
/// <paramref name="other" summary="The other object it is checking for collissions"/>
/// Checks if the sheet is touching the other object, if so returns true
/// </summary>
bool FlatSheet::isTouching(CollisionObject* other)
{
    // TODO: Implement this function
    return false;
}

/// <summary>
///     Get the closest point on the sheet to the given point
/// </summary>
Vector3 FlatSheet::getClosestPoint(const Vector3& point)
{

    Vector3 norm = getNormalVector();
    float t = (Vector3(0.0f, 0.0f, 0.0f).distance(this->position) -
        point.x * norm.x - point.y * norm.y - this->position.z * point.z)
        / (norm.x * norm.x + norm.y * norm.y + norm.z * norm.z);
    Vector3 closestPoint = {point.x + t * norm.x, point.y + t * norm.y, point.z + t * norm.z};

    if (closestPoint.x > this->topRight.x)
    {
        closestPoint.x = this->topRight.x;
    }
    else if (closestPoint.x < this->topLeft.x)
    {
        closestPoint.x = this->topLeft.x;
    }
    if (closestPoint.y > this->topRight.y)
    {
        closestPoint.y = this->topRight.y;
    }
    else if (closestPoint.y < this->bottomRight.y)
    {
        closestPoint.y = this->bottomRight.y;
    }

    if (closestPoint.z > this->topRight.z)
    {
        closestPoint.z = this->topRight.z;
    }
    else if (closestPoint.z < this->bottomRight.z)
    {
        closestPoint.z = this->bottomRight.z;
    }

    return closestPoint;
}

/// <summary>
/// Get the normal vector of the sheet (a, b, c) in the equation ax + by + cz = d
/// </summary>
Vector3 FlatSheet::getNormalVector() const
{
    return this->rotation.getNormalVector3();
}

/// <summary>
/// Set the normal vector of the sheet
/// </summary>
void FlatSheet::setNormalVector(Vector3* normalVector)
{
    this->normalVector = normalVector;
    this->rotation = Quaternion::fromAxisAngle(Vector3(0, 0, 1), 0);
}

/// <summary>
/// Rotate with the given quaternion and update the corners and normal vector
/// </summary>
void FlatSheet::rotate(Quaternion rotation)
{
    this->rotation = this->rotation * rotation * this->rotation.conjugate();

    this->topRight = rotation.rotate(this->topRight);
    this->topLeft = rotation.rotate(this->topLeft);
    this->bottomRight = rotation.rotate(this->bottomRight);
    this->bottomLeft = rotation.rotate(this->bottomLeft);
    this->normalVector = new Vector3(rotation.rotate(*this->normalVector));

    // Checks to make sure that the corners are correct
    while (this->topRight.x < this->topLeft.x || this->topRight.y < this->bottomRight.y ||
        this->topLeft.y < this->bottomLeft.y || this->bottomRight.x < this->bottomLeft.x)
    {
        if (this->topRight.x < this->topLeft.x)
        {
            Vector3 temp = this->topRight;
            this->topRight = this->topLeft;
            this->topLeft = temp;
        }
        if (this->topRight.y < this->bottomRight.y)
        {
            Vector3 temp = this->topRight;
            this->topRight = this->bottomRight;
            this->bottomRight = temp;
        }
        if (this->topLeft.y < this->bottomLeft.y)
        {
            Vector3 temp = this->topLeft;
            this->topLeft = this->bottomLeft;
            this->bottomLeft = temp;
        }
        if (this->bottomRight.x < this->bottomLeft.x)
        {
            Vector3 temp = this->bottomRight;
            this->bottomRight = this->bottomLeft;
            this->bottomLeft = temp;
        }
    }

}

/// <summary>
/// Rotate about the given axis by the given number of degrees
/// </summary>
void FlatSheet::rotate(float degrees, Vector3 axis)
{
    this->rotate(Quaternion::fromAxisAngle(axis, degrees));
}