cmake_minimum_required(VERSION 3.22)
project(PhysicsEngine LANGUAGES CXX)

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
    enable_language(CUDA)
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

set(LIBRARY_FILES
        include/FullEngineIncludes.h
        include/library.h
        source/library.cpp
        include/General/Vector.h
        source/General/Vector.cpp
        include/General/Quaternion.h
        source/General/Quaternion.cpp
)
add_library(PhysicsEngine STATIC
    ${LIBRARY_FILES}
)

if(HAS_CUDA)
    target_compile_features(PhysicsEngine PUBLIC cxx_std_11)
    set_target_properties(PhysicsEngine
            PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(PhysicsEngine PRIVATE CUDA::cudart)

    # CUDA files and headers
    set(CUDA_CODE

    )

    target_sources(
            PhysicsEngine PRIVATE
        ${CUDA_CODE}
    )
endif()