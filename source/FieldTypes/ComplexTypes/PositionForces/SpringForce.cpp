//
// Created by James Miller on 10/22/2024.
//

#include "../../../../include/FieldTypes/ComplexTypes/PositionForces/SpringForce.h"

void SpringForce::updateForce()
{
    Vector3 displacement = *position - *anchorPoint;
    const float distance = displacement.length();
    const float stretch = distance - restLength;
    displacement = displacement.normalize();
    *force = displacement * (-springConstant * stretch);
}