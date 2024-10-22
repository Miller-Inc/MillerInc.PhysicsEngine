//
// Created by James Miller on 10/22/2024.
//

#include "../../../../include/FieldTypes/ComplexTypes/PositionForces/DragForce.h"

void DragForce::updateForce()
{
    const Vector3 velocity = *position; // Assuming position holds the velocity vector
    const float speed = velocity.length();
    *force = velocity * (-dragCoefficient * speed);
}

