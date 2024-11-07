cmake_minimum_required(VERSION 3.22)
project(PhysicsEngine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23) # C++23

# Set the CMP0146 policy to NEW
if (POLICY CMP0146)
    cmake_policy(SET CMP0146 NEW)
endif()

# Set CUDA policies
if (POLICY CMP0104)
    cmake_policy(SET CMP0104 NEW)
endif()
if (POLICY CMP0105)
    cmake_policy(SET CMP0105 NEW)
endif()

find_package(CUDAToolkit)

if (CUDAToolkit_FOUND)
    message(STATUS "CUDA found")
    include_directories(${CUDAToolkit_INCLUDE_DIRS})
    include_directories(${CUDAToolkit_INCLUDE_DIRS}/include)

    message(STATUS "CUDA include dirs: ${CUDAToolkit_INCLUDE_DIRS}")
    message(STATUS "CUDA libraries: ${CUDAToolkit_LIBRARIES}")
    message(STATUS "CUDA version: ${CUDAToolkit_VERSION}")

    set(HAS_CUDA TRUE)

    if(${CUDAToolkit_VERSION} VERSION_LESS "12.0")
        message(STATUS ${CUDAToolkit_VERSION})
        message(WARNING "This project requires CUDA Toolkit 12.0 or higher.")
        set(HAS_CUDA FALSE)
    else()
        message(STATUS "CUDA Toolkit version is compatible")

        message(STATUS "Enabling CUDA")
        add_definitions(-DCUDA_AVAILABLE=true)
        enable_language(CUDA)
    endif()

else()
    message(STATUS "CUDA not found")
    set(HAS_CUDA FALSE)
endif()

if (HAS_CUDA)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_75,code=sm_75")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_86,code=sm_86")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
    set(CMAKE_CUDA_COMPILE_OBJECT "${CMAKE_CUDA_COMPILE_OBJECT} -gencode arch=compute_75,code=sm_75")
    set(CMAKE_CUDA_COMPILE_OBJECT "${CMAKE_CUDA_COMPILE_OBJECT} -gencode arch=compute_86,code=sm_86")
    set(CMAKE_CUDA_COMPILE_OBJECT "${CMAKE_CUDA_COMPILE_OBJECT} --expt-relaxed-constexpr")
endif()

add_library(PhysicsEngine STATIC
        source/library.cpp
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
        include/CrossPlatformMacros.h
        GPU/GPU-Macros.h
        include/FieldTypes/BaseTypes/Vector3Math.h
        include/FieldTypes/BaseTypes/Vector3Math.h
        source/FieldTypes/BaseTypes/Vector3Math.cpp
        GPU/Cuda-Engine/Math/Vector3Math.cu
)

if(HAS_CUDA)
    target_compile_features(PhysicsEngine PUBLIC cxx_std_11)
    set_target_properties(PhysicsEngine
            PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(PhysicsEngine PRIVATE CUDA::cudart)

    target_sources(
            PhysicsEngine PRIVATE
            GPU/CUDA.Physics/kernel.cu
            GPU/CUDA.Physics/kernel.h
            GPU/Cuda-Engine/gpuScene.h
            GPU/Cuda-Engine/gpuScene.cu
    )
endif()