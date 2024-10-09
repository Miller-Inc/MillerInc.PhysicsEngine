//
// Created by James Miller on 10/8/2024.
//
#include "../../../include/FieldTypes/ComplexTypes/ContactForce.h"

Vector3 ContactForce::get_force() const
{
    return torque();
}