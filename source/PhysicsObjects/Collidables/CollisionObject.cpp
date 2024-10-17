//
// Created by James Miller on 10/8/2024.
//

#include "../../../include/PhysicsObjects/Collidables/CollisionObject.h"

CollisionObject::CollisionObject()
{
    position = Vector3(0, 0, 0);
    velocity = Vector3(0, 0, 0);
    mass = 1.0f;
    angularVelocity = Quaternion(0, 0, 0, 0);
    forces = std::vector<Force>();
    overlappingObjects = std::vector<CollisionObject*>();
    currentMomentum = velocity * mass;
}



Vector3 CollisionObject::getClosestPoint(const Vector3& point)
{
    return position;
}

void CollisionObject::ApplyImpulse(const Vector3& impulse, const Vector3& position)
{
}

void CollisionObject::ApplyAngularImpulse(const Quaternion& impulse)
{
    this->angularVelocity += impulse / mass;
}

void CollisionObject::ApplyImpulse(const Vector3& impulse)
{
    this->velocity += impulse / mass;
}

void CollisionObject::ApplyTorqueImpulse(const Vector3& impulse)
{
}

void CollisionObject::ApplyTorqueImpulse(const Vector3& impulse, const Vector3& position)
{
}

void CollisionObject::ApplyTorqueImpulse(const Vector3& impulse, const Vector3& position, const Vector3& axis)
{
}



/// <summary>
/// Moves from this
/// </summary>
void CollisionObject::step(const float timeStep)
{
    auto acceleration = Vector3(0, 0, 0);
    for (const auto& force : forces)
    {
        acceleration += force.force / mass;
    }

    for (int i = 0; i < forces.size(); i++)
    {
        forces[i].step(timeStep);
        if (forces[i].timeRemaining <= 0 && !forces[i].continuous)
        {
            forces.erase(forces.begin() + i);
            i--; // Decrement i to account for the fact that the vector has been resized
        }
    }

    velocity += acceleration * timeStep;

    position += velocity * timeStep;

}

void CollisionObject::OnSeparation(CollisionObject* other)
{
    for (int i = 0; i < overlappingObjects.size(); i++)
    {
        if (overlappingObjects[i] == other)
        {
            overlappingObjects.erase(overlappingObjects.begin() + i);
            i--; // Decrement i to account for the fact that the vector has been resized
        }
    }
}

void CollisionObject::OnContact(CollisionObject* other)
{
    overlappingObjects.push_back(other);
}

void CollisionObject::OnCollision(CollisionObject* other)
{
    this->overlappingObjects.push_back(other);
}

bool CollisionObject::isColliding(CollisionObject* other)
{
    return (other->position == this->position);
}

bool CollisionObject::isTouching(CollisionObject* other)
{
    return (other->position == this->position);
}

void CollisionObject::rotate(const Quaternion rotation)
{
    this->rotation = this->rotation * rotation * this->rotation.conjugate();
}

void CollisionObject::rotate(const float degrees, const Vector3 axis)
{
    this->rotate(Quaternion::fromAxisAngle(axis, degrees));
}

std::string CollisionObject::toString()
{
    return "CollisionObject: Position: " + position.toString() +
        "; Rotation: " + rotation.toString();
}