//
// Created by James Miller on 3/28/2025.
//

#include "../../../include/BasePhysics/PhysicsSim/BasePhysicsFunctions.h"

#include "../../../include/BasePhysics/MObject.h"
#include "../../../include/BasePhysics/MBounds.h"
#include "../../../include/GeneralTypes.h"

namespace MillerPhysics::PhysicsSim
{
    void SimplePhysicsSimulation(MObject& obj, float secondsElapsed)
    {
        // Simple physics simulation
        // Apply gravity to the object
        const auto gravity = MVector(0, 0, -1 * CalculateGravityAcceleration(obj.GetPosition()));
        obj.SetPosition(obj.GetPosition() + obj.GetMass() * gravity * secondsElapsed);
    }

    float CalculateGravityAcceleration(const MVector& pos)
    {
        return M_UNIVERSAL_GRAVITATION * (M_MASS_OF_EARTH /
            ((pos.Length() + M_RADIUS_OF_EARTH) * (pos.Length() + M_RADIUS_OF_EARTH)));
    }
}