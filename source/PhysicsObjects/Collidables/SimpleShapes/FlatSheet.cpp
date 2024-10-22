//
// Created by James Miller on 10/9/2024.
//

#include "../../../../include/PhysicsObjects/Collidables/SimpleShapes/FlatSheet.h"

#include <cmath>
#include <iostream>

FlatSheet::FlatSheet() : FlatSheet::FlatSheet(1.0f, 1.0f)
{
    // Default Constructor
}

FlatSheet::FlatSheet(float sideLength) : FlatSheet::FlatSheet(sideLength, sideLength)
{
    // Constructor
}

FlatSheet::FlatSheet(float height, float width)
: FlatSheet(height, width, new Vector3(0, 0, 0), 1.0f)
{
    // Constructor
}

FlatSheet::FlatSheet(float height, float width, Vector3* position, float mass) : CollisionObject()
{
    // Constructor
    this->height = height;
    this->width = width;
    this->position = position;
    this->mass = mass;

    // Set the corners of the sheet
    this->topRight = new Vector3(width / 2, height / 2, 0);
    this->topLeft = new Vector3(-width / 2, height / 2, 0);
    this->bottomRight = new Vector3(width / 2, -height / 2, 0);
    this->bottomLeft = new Vector3(-width / 2, -height / 2, 0);

    // Set the rotation of the sheet
    this->normalVector = new Vector3(0, 0, 1);
    this->rotation = new Quaternion(0, 0, 0, 1);


    this->name = "Flat Sheet " + std::to_string(fSCount());
    incrementFS();
}

FlatSheet::~FlatSheet() = default;

/// <summary>
/// Checks to see if the sheet is colliding with the other object
/// </summary>
bool FlatSheet::isColliding(CollisionObject* other)
{
    Vector3 thisClosestPoint = this->getClosestPoint(*other->position);
    Vector3 otherClosestPoint = other->getClosestPoint(thisClosestPoint);

    std::cout << "This Closest Point: " << thisClosestPoint.toString() << "\n";

    std::cout << "Other Closest Point: " << otherClosestPoint.toString() << "\n";

    if (std::fabs(other->position->x) > std::fabs(thisClosestPoint.x) &&
        std::fabs(other->position->y) > std::fabs(thisClosestPoint.y) &&
        std::fabs(other->position->z) > std::fabs(thisClosestPoint.z) &&
        std::fabs(otherClosestPoint.x) < std::fabs(this->position->x) &&
        std::fabs(otherClosestPoint.y) < std::fabs(this->position->y) &&
        std::fabs(otherClosestPoint.z) < std::fabs(this->position->z))
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
Vector3 FlatSheet::getClosestPoint(const Vector3& point) {
    Vector3 norm = this->getNormalVector();
    float normSquared = norm.x * norm.x + norm.y * norm.y + norm.z * norm.z;

    // Avoid division by zero
    if (normSquared == 0) {
        return {0.0f, 0.0f, 0.0f};
    }

    float t = (this->position->dot(norm) - point.dot(norm)) / normSquared;
    Vector3 closestPoint = point + norm * t;

    // Clamp the closest point to the bounds of the flat sheet
    closestPoint.x = std::clamp(closestPoint.x, this->topLeft->x, this->topRight->x);
    closestPoint.y = std::clamp(closestPoint.y, this->bottomLeft->y, this->topLeft->y);
    closestPoint.z = std::clamp(closestPoint.z, this->bottomLeft->z, this->topRight->z);

    return closestPoint;
}

/// <summary>
/// Get the normal vector of the sheet (a, b, c) in the equation ax + by + cz = d
/// </summary>
Vector3 FlatSheet::getNormalVector() const
{
    return this->rotation->getNormalVector3();
}

/// <summary>
/// Set the normal vector of the sheet
/// </summary>
void FlatSheet::setNormalVector(Vector3* normalVector)
{
    this->normalVector = normalVector;
    this->rotation = new Quaternion(Quaternion::fromAxisAngle(Vector3(0, 0, 1), 0));
}

/// <summary>
/// Rotate with the given quaternion and update the corners and normal vector
/// </summary>
void FlatSheet::rotate(Quaternion rotation)
{
    this->rotation = new Quaternion(*this->rotation * rotation * this->rotation->conjugate());

    this->topRight = rotation.rotate(this->topRight);
    this->topLeft = rotation.rotate(this->topLeft);
    this->bottomRight = rotation.rotate(this->bottomRight);
    this->bottomLeft = rotation.rotate(this->bottomLeft);
    this->normalVector = new Vector3(rotation.rotate(*this->normalVector));

    // Checks to make sure that the corners are correct
    while (this->topRight->x < this->topLeft->x || this->topRight->y < this->bottomRight->y ||
        this->topLeft->y < this->bottomLeft->y || this->bottomRight->x < this->bottomLeft->x)
    {
        if (this->topRight->x < this->topLeft->x)
        {
            Vector3* temp = this->topRight;
            this->topRight = this->topLeft;
            this->topLeft = temp;
        }
        if (this->topRight->y < this->bottomRight->y)
        {
            Vector3* temp = this->topRight;
            this->topRight = this->bottomRight;
            this->bottomRight = temp;
        }
        if (this->topLeft->y < this->bottomLeft->y)
        {
            Vector3* temp = this->topLeft;
            this->topLeft = this->bottomLeft;
            this->bottomLeft = temp;
        }
        if (this->bottomRight->x < this->bottomLeft->x)
        {
            Vector3* temp = this->bottomRight;
            this->bottomRight = this->bottomLeft;
            this->bottomLeft = temp;
        }
    }

}

//// <summary>
///      Rotate about the given axis by the given number of degrees
/// </summary>
void FlatSheet::rotate(float degrees, Vector3 axis)
{
    this->rotate(Quaternion::fromAxisAngle(axis, degrees));
}

/// <summary>
///     Returns the string representation of this object
/// </summary>
std::string FlatSheet::toString()
{
    return this->name + " at " + this->position->toString() + " with a height of " + std::to_string(this->height) +
        " meters and a width of " + std::to_string(this->width) + " meters and a mass of " + std::to_string(this->mass)
        + " kg";
}

int FlatSheet::fSCount()
{
    return fSCounter;
}

void FlatSheet::incrementFS()
{
    fSCounter++;
}

int FlatSheet::fSCounter = 0;

Vector3* FlatSheet::getClosestPoint(const Vector3* point)
{
    return new Vector3(this->getClosestPoint(*point));
}