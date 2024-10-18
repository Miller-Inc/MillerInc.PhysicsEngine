//
// Created by James Miller on 10/8/2024.
//

#include "../../include/PhysicsObjects/BaseObject.h"

BaseObject::BaseObject() : BaseObject(new Vector3(0, 0, 0), new Vector3(0, 0, 0), new Quaternion(0, 0, 0, 0), 1.0f)
{

}

BaseObject::BaseObject(Vector3* position, Vector3* velocity, Quaternion* rotation, float mass)
{
    this->position = position;
    this->velocity = velocity;
    this->rotation = rotation;
    this->mass = mass;
    this->angularVelocity = new Quaternion(0, 0, 0, 0);
    this->rotation = new Quaternion(0, 0, 0, 0);
}


void BaseObject::step( float timeStep)
{
    const Vector3 v (*velocity * timeStep);
    position->x += v.x;
    position->y += v.y;
    position->z += v.z;
}
