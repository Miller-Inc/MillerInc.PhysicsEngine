//
// Created by James Miller on 10/9/2024.
//

#include "../../../../include/PhysicsObjects/Collidables/SimpleShapes/Cube.h"

#include <iostream>

Cube::Cube() : Cube::Cube(1.0f)
{
    // Default Constructor
}

Cube::Cube(float sideLength) : CollisionObject()
{
    // Constructor
    this->sideLength = sideLength;
    this->top = new FlatSheet(sideLength, sideLength, new Vector3(0, sideLength / 2, 0), 1.0f);
    this->bottom = new FlatSheet(sideLength, sideLength, new Vector3(0, -sideLength / 2, 0), 1.0f);
    this->left = new FlatSheet(sideLength, sideLength, new Vector3(-sideLength / 2, 0, 0), 1.0f);
    this->right = new FlatSheet(sideLength, sideLength, new Vector3(sideLength / 2, 0, 0), 1.0f);
    this->front = new FlatSheet(sideLength, sideLength, new Vector3(0, 0, sideLength / 2), 1.0f);
    this->back = new FlatSheet(sideLength, sideLength, new Vector3(0, 0, -sideLength / 2), 1.0f);


    this->name = "Cube " + std::to_string(cubeCount());
    incrementCube();
}

Cube::~Cube() = default;

bool Cube::isColliding(CollisionObject* other)
{
    Vector3 thisClosestPoint = this->getClosestPoint(*other->position);
    Vector3 otherClosestPoint = other->getClosestPoint(thisClosestPoint);
    thisClosestPoint = this->getClosestPoint(otherClosestPoint);

    if (otherClosestPoint.distance(position) < sideLength / 2)
    {
        return true;
    }

    FlatSheet* closest = this->getClosestPlane(otherClosestPoint);

    std::cout << "This Closest Point: " << thisClosestPoint.toString() << "\n";
    std::cout << "Other Closest Point: " << otherClosestPoint.toString() << "\n";

    if (closest->position->distance(position) < otherClosestPoint.distance(position))
    {
        return closest->isColliding(other);
    }

    return false;
}

bool Cube::isTouching(CollisionObject* other)
{
    if (this->isColliding(other))
    {
        return true;
    }
    return false;
}

/// <summary>
/// Gets the closest point on the cube to the given point
/// </summary>
Vector3 Cube::getClosestPoint(const Vector3& point)
{
    return this->getClosestPlane(point)->getClosestPoint(point);
}

Quaternion Cube::getRotation() const
{
    return front->rotation->getNormalVector();
}

void Cube::rotate(Quaternion rotation)
{
    top->rotate(rotation);
    bottom->rotate(rotation);
    left->rotate(rotation);
    right->rotate(rotation);
    front->rotate(rotation);
    back->rotate(rotation);
}

void Cube::rotate(float degrees, Vector3 axis)
{
    this->rotate(Quaternion::fromAxisAngle(axis, degrees));
}

std::string Cube::toString()
{
    return "Cube";
}

int Cube::cubeCount()
{
    return cubeCounter;
}

void Cube::incrementCube()
{
    cubeCounter++;
}

void Cube::step(float timeStep)
{
    top->step(timeStep);
    bottom->step(timeStep);
    left->step(timeStep);
    right->step(timeStep);
    front->step(timeStep);
    back->step(timeStep);

    CollisionObject::step(timeStep);
}

int Cube::cubeCounter = 0;

FlatSheet* Cube::getClosestPlane(const Vector3& point) const
{
    Vector3 cP = top->getClosestPoint(point);
    FlatSheet* closest = top;
    if (bottom->getClosestPoint(point).distance(point) < cP.distance(point))
    {
        cP = bottom->getClosestPoint(point);
        closest = bottom;
    }
    if (left->getClosestPoint(point).distance(point) < cP.distance(point))
    {
        cP = left->getClosestPoint(point);
        closest = left;
    }
    if (right->getClosestPoint(point).distance(point) < cP.distance(point))
    {
        cP = right->getClosestPoint(point);
        closest = right;
    }
    if (front->getClosestPoint(point).distance(point) < cP.distance(point))
    {
        cP = front->getClosestPoint(point);
        closest = front;
    }
    if (back->getClosestPoint(point).distance(point) < cP.distance(point))
    {
        closest = back;
    }

    return closest;
}