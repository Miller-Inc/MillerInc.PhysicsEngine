//
// Created by James Miller on 10/8/2024.
//

#include "../../../../include/PhysicsObjects/Collidables/SimpleShapes/Sphere.h"

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
    this->position = position;
    this->velocity = velocity;
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
    // Implementation
}

void Sphere::ApplyImpulse(const Vector3& impulse, const Vector3& position)
{
    // Implementation
}

void Sphere::ApplyAngularImpulse(const Quaternion& impulse)
{
    // Implementation
}

void Sphere::ApplyTorqueImpulse(const Vector3& impulse)
{
    // Implementation
}

void Sphere::ApplyTorqueImpulse(const Vector3& impulse, const Vector3& position)
{
    // Implementation
}

void Sphere::ApplyTorqueImpulse(const Vector3& impulse, const Vector3& position, const Vector3& axis)
{
    // Implementation
}

void Sphere::OnCollision(CollisionObject* other)
{
    // Implementation
}

void Sphere::OnSeparation(CollisionObject* other)
{
    // Implementation
}

void Sphere::OnContact(CollisionObject* other)
{
    // Implementation
}

std::string Sphere::toString()
{
    return this->name + " position: " + position.toString() + " Velocity: " +
        velocity.toString() + " Mass: " + std::to_string(mass) + " Radius " + std::to_string(radius);
}

bool Sphere::isColliding(CollisionObject* other)
{
    if (other->getClosestPoint(this->position).distance(this->position) < this->radius)
    {
        return true;
    }
    return false;
}

bool Sphere::isTouching(CollisionObject* other)
{
    return Sphere::isColliding(other);
}

Vector3 Sphere::getClosestPoint(const Vector3& point)
{
    // Step 1: Calculate the direction vector from the sphere's center to the point
    Vector3 direction = point - this->position;

    // Step 2: Normalize the direction vector
    direction = direction / direction.magnitude();

    // Step 3: Scale the direction vector by the sphere's radius
    direction = direction * this->radius;

    // Step 4: Add the scaled vector to the sphere's center to get the closest point
    Vector3 closestPoint = this->position + direction;

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
