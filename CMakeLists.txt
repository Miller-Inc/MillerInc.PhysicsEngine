cmake_minimum_required(VERSION 3.22)
project(PhysicsEngine)

set(CMAKE_CXX_STANDARD 23)

#find_package(Qt6 REQUIRED COMPONENTS Widgets)

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
        source/PhysicsEngine/Scenes/Scene.cpp
        source/Rendering/Windows/CPU/visualizer.cpp
        include/Rendering/Windows/CPU/visualizer.h
        include/Rendering/Windows/CPU/visualizer.ui
        include/PhysicsObjects/Collidables/SimpleShapes/Cube.h
        source/PhysicsObjects/Collidables/SimpleShapes/Cube.cpp
        include/PhysicsObjects/Collidables/SimpleShapes/FlatSheet.h
        source/PhysicsObjects/Collidables/SimpleShapes/FlatSheet.cpp
        source/FieldTypes/BaseTypes/Quaternion.cpp
        include/FieldTypes/ComplexTypes/PosVariableForce.h
        include/FieldTypes/ComplexTypes/PositionForces/GravityForce.h
        source/FieldTypes/ComplexTypes/PositionForces/GravityForce.cpp
        include/FieldTypes/ComplexTypes/PositionForces/SpringForce.h
        include/FieldTypes/ComplexTypes/PositionForces/DragForce.h
        source/FieldTypes/ComplexTypes/PositionForces/DragForce.cpp
        source/FieldTypes/ComplexTypes/PositionForces/SpringForce.cpp
        include/Rendering/Shared/GPU/NVIDIA/GPUScene.h
        source/Rendering/Shared/GPU/NVIDIA/GPUScene.cpp
        include/Rendering/Shared/GPU/NVIDIA/gpu_kernels.cu
)
