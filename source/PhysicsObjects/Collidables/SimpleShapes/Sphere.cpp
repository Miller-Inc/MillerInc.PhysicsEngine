//
// Created by James Miller on 10/8/2024.
//

#include "../../../../include/PhysicsObjects/Collidables/SimpleShapes/Sphere.h"

#include <iostream>

// Constructors
Sphere::Sphere() : Sphere(1.0f)
{

}

Sphere::Sphere(float radius) : Sphere(radius, 1.0f)
{

}

Sphere::Sphere(float radius, float mass) : Sphere(radius, mass, Vector3(0, 0, 0))
{

}

Sphere::Sphere(float radius, float mass, const Vector3& position) : Sphere(radius, mass, position, Vector3(0, 0, 0))
{

}

/// <summary> Main Constructor </summary>
Sphere::Sphere(float radius, float mass, const Vector3& position, const Vector3& velocity)
{
    this->friction = 0.0f;
    this->useFriction = false;
    this->radius = radius;
    this->mass = mass;
    this->position = new Vector3(position);
    this->velocity = new Vector3(velocity);
    this->forces = std::vector<Force>();
    this->name = "Sphere " + std::to_string(sphereCount());
    incrementSpheres();
}

Sphere::Sphere(float radius, float mass, float friction, const Vector3& position) : Sphere(radius, mass, position, Vector3(0, 0, 0))
{
    this->friction = friction;
}

// Methods
void Sphere::ApplyImpulse(const Vector3& impulse)
{
    ApplyImpulse(impulse, *position);
}

void Sphere::ApplyImpulse(const Vector3& impulse, const Vector3& position)
{
    this->velocity = new Vector3(*velocity + impulse / mass);
    this->currentMomentum = new Vector3(velocity->x * mass, velocity->y * mass,
        velocity->z * mass);

    // TODO: Finish this method
}

void Sphere::ApplyAngularImpulse(const Quaternion& impulse)
{
    // Implementation
    //  TODO: Implement this method
}

void Sphere::ApplyTorqueImpulse(const Vector3& impulse)
{
    // Implementation
    // TODO: Implement this method
}

void Sphere::ApplyTorqueImpulse(const Vector3& impulse, const Vector3& position)
{
    // Implementation
    // TODO: Implement this method
}

void Sphere::ApplyTorqueImpulse(const Vector3& impulse, const Vector3& position, const Vector3& axis)
{
    // Implementation
    // TODO: Implement this method
}

void Sphere::OnCollision(CollisionObject* other)
{
    // Implementation
    // TODO: Implement this method
}

void Sphere::OnSeparation(CollisionObject* other)
{
    // Implementation
    // TODO: Implement this method
}

void Sphere::OnContact(CollisionObject* other)
{
    // Implementation
    // TODO: Implement this method
}

/// <summary>
///     Gets the string representation of the sphere
/// </summary>
std::string Sphere::toString()
{
    return this->name + " position: " + position->toString() + " Velocity: " +
        velocity->toString() + " Mass: " + std::to_string(mass) + " Radius " + std::to_string(radius);
}

/// <summary>
///     Check if the sphere is colliding with the other object
/// </summary>
bool Sphere::isColliding(CollisionObject* other)
{
    Vector3 cP = this->getClosestPoint(*other->position);
    Vector3 oCP = other->getClosestPoint(cP);
    cP = this->getClosestPoint(oCP);
    oCP = other->getClosestPoint(cP);

    std::cout << "Sphere: " << this->name << " Closest Point: " << cP.toString() << std::endl;
    std::cout << "Other: " << other->name << " Closest Point: " << oCP.toString() << std::endl;

    if ((cP - oCP).magnitude() < this->radius)
    {
        return true;
    }
    return false;
}

/// <summary>
///   Check if the sphere is touching the other object
/// </summary>
bool Sphere::isTouching(CollisionObject* other)
{
    return Sphere::isColliding(other);
}

/// <summary>
///    Get the closest point on the sphere's surface to the given point
/// </summary>
Vector3 Sphere::getClosestPoint(const Vector3& point)
{
    // Step 1: Calculate the direction vector from the sphere's center to the point
    auto direction = Vector3(point - this->position);

    if (direction.magnitude() == 0)
    {
        return point;
    }

    // Step 2: Normalize the direction vector
    direction = direction / direction.magnitude();

    // Step 3: Scale the direction vector by the sphere's radius
    direction = direction * this->radius;

    // Step 4: Add the scaled vector to the sphere's center to get the closest point
    Vector3 closestPoint = *this->position + direction;

    return closestPoint;
}

int Sphere::sphereCount()
{
    return sphereCounter;
}

void Sphere::incrementSpheres()
{
    sphereCounter++;
}

int Sphere::sphereCounter = 0;
