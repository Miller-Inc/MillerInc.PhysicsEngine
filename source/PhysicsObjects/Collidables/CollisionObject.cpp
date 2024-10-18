//
// Created by James Miller on 10/8/2024.
//

#include "../../../include/PhysicsObjects/Collidables/CollisionObject.h"

CollisionObject::CollisionObject()
{
    position = new Vector3(0, 0, 0);
    velocity = new Vector3(0, 0, 0);
    mass = 1.0f;
    angularVelocity = new Quaternion(0, 0, 0, 0);
    forces = std::vector<Force>();
    overlappingObjects = std::vector<CollisionObject*>();
    currentMomentum = new Vector3(velocity->x * mass, velocity->y * mass, velocity->z * mass);
    this->name = name + std::to_string(collObjCount());
    incrementCollisionObjs();
}



Vector3 CollisionObject::getClosestPoint(const Vector3& point)
{
    return *position;
}

Vector3* CollisionObject::getClosestPoint(const Vector3* point)
{
    return position;
}

void CollisionObject::ApplyImpulse(const Vector3& impulse, const Vector3& position)
{
}

void CollisionObject::ApplyImpulse(const Vector3* impulse)
{
}

void CollisionObject::ApplyAngularImpulse(const Quaternion& impulse)
{
    this->angularVelocity = new Quaternion(*this->angularVelocity + impulse / mass);
}

void CollisionObject::ApplyImpulse(const Vector3& impulse)
{
    this->velocity = new Vector3(*this->velocity + impulse / mass);
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
        acceleration += *force.force / mass;
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

    velocity = new Vector3(*velocity + Vector3(acceleration * timeStep));

    // Update position based on velocity and time step
    this->position = new Vector3(*this->position + *this->velocity * timeStep);

    // Update rotation based on angular velocity and time step
    this->rotation = new Quaternion(*this->rotation * *this->angularVelocity  * this->rotation->conjugate() * timeStep);

    // Normalize the rotation quaternion
    this->rotation = this->rotation->normalizeP();

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
    this->rotation = new Quaternion(*this->rotation * rotation * this->rotation->conjugate());
}

void CollisionObject::rotate(const float degrees, const Vector3 axis)
{
    this->rotate(Quaternion::fromAxisAngle(axis, degrees));
}

std::string CollisionObject::toString()
{
    return this->name + " position: " + position->toString() + " Velocity: " +
        velocity->toString() + " Mass: " + std::to_string(mass);
}

int CollisionObject::collObjCount()
{
    return objCounter;
}

void CollisionObject::incrementCollisionObjs()
{
    objCounter++;
}

int CollisionObject::objCounter = 0;

bool CollisionObject::equals(BaseObject* other)
{
    auto otherCollisionObject = dynamic_cast<CollisionObject*>(other);
    if (otherCollisionObject == nullptr)
    {
        return false;
    }
    return this->position == otherCollisionObject->position &&
        this->velocity == otherCollisionObject->velocity &&
        this->rotation == otherCollisionObject->rotation &&
        this->angularVelocity == otherCollisionObject->angularVelocity &&
        this->mass == otherCollisionObject->mass && this->name == otherCollisionObject->name;
}