//
// Created by James Miller on 10/23/2024.
//

#ifndef GPUSCENE_H
#define GPUSCENE_H

#include <vector>
#include <cuda_runtime.h>
#include "../../../../FieldTypes/BaseTypes/Vector3.h"
#include "../../../../PhysicsObjects/Collidables/CollisionObject.h"

class GPUScene {
public:
    GPUScene();
    ~GPUScene();

    void addObject(CollisionObject* object);
    void step(float timestep);

private:
    std::vector<CollisionObject*> objects;
    float* d_positions;
    float* d_velocities;
    int numObjects;

    void allocateMemory();
    void freeMemory();
    void copyDataToDevice();
    void copyDataToHost();
};

#endif // GPUSCENE_H
