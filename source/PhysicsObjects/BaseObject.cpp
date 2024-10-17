//
// Created by James Miller on 10/8/2024.
//

#include "../../include/PhysicsObjects/BaseObject.h"

BaseObject::BaseObject() : BaseObject(Vector3(0, 0, 0), Vector3(0, 0, 0), Quaternion(0, 0, 0, 0), 1.0f)
{

}

BaseObject::BaseObject(Vector3 position, Vector3 velocity, Quaternion rotation, float mass): rotation(0, 0, 0, 0), angularVelocity(0, 0, 0, 0)
{
    this->position = position;
    this->velocity = velocity;
    this->rotation = rotation;
    this->mass = mass;
    this->angularVelocity = Quaternion(0, 0, 0, 0);
    this->rotation = Quaternion(0, 0, 0, 0);
}


void BaseObject::step(const float timeStep)
{
    position += velocity * timeStep;
}
