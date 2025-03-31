//
// Created by James Miller on 3/28/2025.
//

#pragma once
#include "../../CrossPlatformMacros.h"
#include "../../GeneralTypes.h"

namespace MillerPhysics::PhysicsSim
{
    void SimplePhysicsSimulation(MObject& obj, float secondsElapsed);

    float CalculateGravityAcceleration(const MVector& pos);
}