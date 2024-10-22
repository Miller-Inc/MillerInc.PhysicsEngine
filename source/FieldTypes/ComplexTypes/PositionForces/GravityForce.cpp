//
// Created by James Miller on 10/21/2024.
//

#include "../../../../include/FieldTypes/ComplexTypes/PositionForces/GravityForce.h"

void GravityForce::updateForce()
{
    const Vector3 g = GRAVITATIONALCONSTANT();
    const float distance = position->length();
    force = new Vector3(g.x * (mass1 * mass2) / (distance * distance),
                     g.y * (mass1 * mass2) / (distance * distance),
                     g.z * (mass1 * mass2) / (distance * distance));
}

