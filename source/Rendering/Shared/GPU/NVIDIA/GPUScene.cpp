//
// Created by James Miller on 10/23/2024.
//

#include "../../../../../include/Rendering/Shared/GPU/NVIDIA/GPUScene.h"
#include "../../../../../include/Rendering/Shared/GPU/NVIDIA/gpu_kernels.h"

GPUScene::GPUScene() : d_positions(nullptr), d_velocities(nullptr), numObjects(0) {}

GPUScene::~GPUScene() {
    freeMemory();
}

void GPUScene::addObject(CollisionObject* object) {
    objects.push_back(object);
    numObjects = objects.size();
    allocateMemory();
    copyDataToDevice();
}

void GPUScene::step(float timestep) {
    copyDataToDevice();
    launchUpdatePositions(d_positions, d_velocities, numObjects, timestep);
    copyDataToHost();
}

void GPUScene::allocateMemory() {
    cudaFree(d_positions);
    cudaFree(d_velocities);
    cudaMalloc(&d_positions, numObjects * 3 * sizeof(float));
    cudaMalloc(&d_velocities, numObjects * 3 * sizeof(float));
}

void GPUScene::freeMemory() {
    cudaFree(d_positions);
    cudaFree(d_velocities);
}

void GPUScene::copyDataToDevice() {
    std::vector<float> h_positions(numObjects * 3);
    std::vector<float> h_velocities(numObjects * 3);
    for (int i = 0; i < numObjects; ++i) {
        h_positions[i * 3] = objects[i]->position->x;
        h_positions[i * 3 + 1] = objects[i]->position->y;
        h_positions[i * 3 + 2] = objects[i]->position->z;
        h_velocities[i * 3] = objects[i]->velocity->x;
        h_velocities[i * 3 + 1] = objects[i]->velocity->y;
        h_velocities[i * 3 + 2] = objects[i]->velocity->z;
    }
    cudaMemcpy(d_positions, h_positions.data(), numObjects * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_velocities, h_velocities.data(), numObjects * 3 * sizeof(float), cudaMemcpyHostToDevice);
}

void GPUScene::copyDataToHost() {
    std::vector<float> h_positions(numObjects * 3);
    cudaMemcpy(h_positions.data(), d_positions, numObjects * 3 * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < numObjects; ++i) {
        objects[i]->position->x = h_positions[i * 3];
        objects[i]->position->y = h_positions[i * 3 + 1];
        objects[i]->position->z = h_positions[i * 3 + 2];
    }
}