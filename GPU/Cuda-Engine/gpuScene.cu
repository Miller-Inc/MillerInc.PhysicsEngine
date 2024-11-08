#if CUDA_AVAILABLE

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../../include/PhysicsEngine/Scenes/Scene.h"
#include "gpuScene.h"

__global__ void processSceneKernel(Scene* scene, int sceneSize) {
    int i = threadIdx.x;
    if (i < sceneSize) {
        // Example processing: move each object along the x-axis
    }
}

void GPUScene::runSceneOnGPU(Scene* scene) {
    Scene* dev_scene = nullptr;
    int sceneSize = scene->sceneObjects.size();
    cudaMalloc((void**)&dev_scene, sizeof(Scene));
    cudaMemcpy(dev_scene, scene, sizeof(Scene), cudaMemcpyHostToDevice);

    processSceneKernel<<<1, sceneSize>>>(dev_scene, sceneSize);

    cudaMemcpy(scene, dev_scene, sizeof(Scene), cudaMemcpyDeviceToHost);
    cudaFree(dev_scene);
}

#endif