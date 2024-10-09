cmake_minimum_required(VERSION 3.29)
project(PhysicsEngine)

set(CMAKE_CXX_STANDARD 26)

add_library(PhysicsEngine STATIC source/library.cpp
        include/library.h
        include/FullEngineIncludes.h
        include/FieldTypes/BaseTypes/Vector3.h
        source/FieldTypes/BaseTypes/Vector3.cpp
        include/FieldTypes/BaseTypes/Quaternion.h
        include/FieldTypes/ComplexTypes/Force.h
        source/FieldTypes/ComplexTypes/Force.cpp
        include/FieldTypes/ComplexTypes/ContactForce.h
        source/FieldTypes/ComplexTypes/ContactForce.cpp
        include/PhysicsObjects/BaseObject.h
        source/PhysicsObjects/BaseObject.cpp
        source/PhysicsObjects/Collidables/ImpulseObject.cpp
        include/PhysicsObjects/Collidables/ImpulseObject.h
        include/PhysicsObjects/Collidables/CollisionObject.h
        source/PhysicsObjects/Collidables/CollisionObject.cpp
        include/PhysicsObjects/Collidables/SimpleShapes/Sphere.h
        source/PhysicsObjects/Collidables/SimpleShapes/Sphere.cpp
        include/PhysicsEngine/Scenes/Scene.h
        source/PhysicsEngine/Scenes/Scene.cpp)
