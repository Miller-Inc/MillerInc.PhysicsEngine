//
// Created by James Miller on 10/8/2024.
//

#pragma once

#ifndef FULLENGINEINCLUDES_H
#define FULLENGINEINCLUDES_H

// Include all headers here

// Base Types
#include "FieldTypes/BaseTypes/Vector3.h"
#include "FieldTypes/BaseTypes/Quaternion.h"
#include "FieldTypes/BaseTypes/Vector3Math.h"

// Complex Types
#include "FieldTypes/ComplexTypes/Force.h"
#include "FieldTypes/ComplexTypes/ContactForce.h"
#include "FieldTypes/ComplexTypes/PosVariableForce.h"
    // Position Forces
    #include "FieldTypes/ComplexTypes/PositionForces/GravityForce.h"
    #include "FieldTypes/ComplexTypes/PositionForces/SpringForce.h"
    #include "FieldTypes/ComplexTypes/PositionForces/DragForce.h"

// Physics Engine
    // Object Types
    #include "PhysicsObjects/BaseObject.h"

        // Collidable Types
        #include "PhysicsObjects/Collidables/ImpulseObject.h"
        #include "PhysicsObjects/Collidables/CollisionObject.h"

            // Simple Shapes
            #include "PhysicsObjects/Collidables/SimpleShapes/Sphere.h"
            #include "PhysicsObjects/Collidables/SimpleShapes/Cube.h"
            #include "PhysicsObjects/Collidables/SimpleShapes/FlatSheet.h"

    // Scene Types
    #include "PhysicsEngine/Scenes/Scene.h"

#endif //FULLENGINEINCLUDES_H
