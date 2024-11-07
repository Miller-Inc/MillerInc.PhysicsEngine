# Miller Inc Physics Engine
This is a simple physics engine library that contains all the logic for 
a realistic physics simulation. This library is written in C++ and has 
header includes, so you can include the library in your project by 
including the header files in your project.

## Features
- 2D Physics Simulation
- 3D Physics Simulation
- Collision Detection
- Collision Resolution
- Rigid Body Dynamics
- Particle Dynamics
- Force Generators
- And many other features that have yet to be developed or implemented

Note that not all of these features are implemented yet, but they are planned

## Installation
To install the library, you can download the repository and add the library to your project.
### To Use in C++ Projects
1. Clone the repository
2. Build this project with CMAKE
   1. Enter the repository directory and create a build directory
       ~~~ shell 
            cd MillerInc.PhysicsEngine
            mkdir build
            cd build
       ~~~
   2. Run the following commands to build the project:
      ~~~ shell
      cmake ..
      make
      ~~~
      This will result in a library being built in the `build` directory with the name `libPhysicsEngine.a`
   3. You can now include the library in your project by including the header files in your project as well as the library
   4. To include the library in your project, you can add the following line to your CMakeLists.txt file:
      ~~~ CMAKE
      target_link_libraries(your_project_name /path/to/MillerInc.PhysicsEngine/build/libPhysicsEngine.a)
      add_subdirectory(/path/to/MillerInc.PhysicsEngine/include)
      ~~~
      Replace `your_project_name` with the name of your project and `/path/to/MillerInc.PhysicsEngine/build/libPhysicsEngine.a` with the path to the library you built
3. Add the `include` folder to your project directory
4. If you have a GPU with CUDA support, you should copy the GPU directory 
to your project directory as well and build this repository with the GPU enabled
5. Include the header files in your project, simple include statement:
   ~~~ C++
        #include "include/FullEngineIncludes.h"
   ~~~
6. Reference any classes or functions from the library in your project

### For Use in Console, Desktop, or Mobile Applications
1. Clone the repository
2. Add the `include` folder to your project
3. Build the project with the library included for the platform you are developing for
4. Reference any classes or functions from the library in your project

## Notes
1. This library is still in development and is not yet ready for production use.
Please use at your own risk.
2. This library is licensed under the GNU GENERAL PUBLIC License. You can view the license [here](https://github.com/Miller-Inc/MillerInc.PhysicsEngine/blob/master/LICENSE.txt)
3. If you have any questions or comments, please feel free to contact me
4. This is a library, not a standalone application. You will need to include the library in your project to use it.


## Project Goals 
1. Implement all features listed in the features section
2. Implement a simple physics simulation using the library
3. Implement a simple 3D physics simulation using the library
4. Implement a simple collision detection system
5. Implement a simple collision resolution
6. Start implementing a basic visual representation of the physics simulation
7. Accelerate the processing speed of the physics simulation by using multi-threading and even GPU acceleration

## Contributing
If you would like to contribute to the project, you can fork the repository and submit a pull request with your changes.
Let me know what you are developing and I will be happy to help you with your project.