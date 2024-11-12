#if CUDA_AVAILABLE

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../../include/PhysicsEngine/Scenes/Scene.h"
#include "gpuScene.h"

__device__ int dev_removeIndex;

__global__ void findCollisionObject(CollisionObject** collision_objects, int numCollisionObjects, CollisionObject* obj) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < numCollisionObjects) {
        bool result = collision_objects[i]->position == obj->position &&
            collision_objects[i]->velocity == obj->velocity &&
            collision_objects[i]->rotation == obj->rotation &&
            collision_objects[i]->angularVelocity == obj->angularVelocity &&
            collision_objects[i]->mass == obj->mass /*&&
            strcmp(collision_objects[i]->name, obj->name) == 0*/;

        if (result) {
            dev_removeIndex = i;
        }
    }
}

void GPUScene::removeCollisionObject(CollisionObject* object) {
    int removeIndex = -1;

    std::cout << "Removing object: " << object->name << "\n";

    CollisionObject* dev_object = nullptr;
    cudaMalloc((void**)&dev_object, sizeof(CollisionObject));
    cudaMemcpy(dev_object, object, sizeof(CollisionObject), cudaMemcpyHostToDevice);

    CollisionObject** dev_collisionObjects = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&dev_collisionObjects), numCollisionObjects * sizeof(CollisionObject*));
    cudaMemcpy(dev_collisionObjects, collisionObjects, numCollisionObjects * sizeof(CollisionObject*), cudaMemcpyHostToDevice);

    int numBlocks = (numCollisionObjects + 255) / 256;

    findCollisionObject<<<numBlocks, 256>>>(dev_collisionObjects, numCollisionObjects, dev_object);

    cudaDeviceSynchronize();

    cudaMemcpyFromSymbol(&removeIndex, dev_removeIndex, sizeof(int), 0, cudaMemcpyDeviceToHost);

    std::cout << "Remove index: " << removeIndex << "\n" << std::endl;

    printf(removeIndex == -1 ? "Object not found\n" : "Object found at index %d\n", removeIndex);

    if (removeIndex != -1) {
        delete collisionObjects[removeIndex];
        for (int i = removeIndex; i < numCollisionObjects - 1; i++) {
            collisionObjects[i] = collisionObjects[i + 1];
        }
        numCollisionObjects--;
    }

    cudaFree(dev_object);
    cudaFree(dev_collisionObjects);
}

GPUScene::~GPUScene() {
    for (int i = 0; i < numCollisionObjects; ++i) {
        delete collisionObjects[i];
    }
    delete[] collisionObjects;
}

GPUScene::GPUScene() {
    currentLength = 128;
    numCollisionObjects = 0;
    collisionObjects = new CollisionObject*[currentLength];
    removeIndex = new int;
    *removeIndex = -1;
}

void GPUScene::addCollisionObject(CollisionObject* object) {
    if (numCollisionObjects == currentLength) {
        enlargeCollisionObjectsArray();
    }
    collisionObjects[numCollisionObjects] = object;
    numCollisionObjects++;
}

int GPUScene::getNumCollisionObjects() const {
    return this->numCollisionObjects;
}

void GPUScene::enlargeCollisionObjectsArray() {
    auto** newCollisionObjects = new CollisionObject*[currentLength + 128];
    for (int i = 0; i < numCollisionObjects; ++i) {
        newCollisionObjects[i] = collisionObjects[i];
    }
    delete[] collisionObjects;
    collisionObjects = newCollisionObjects;
    currentLength += 128;
}

void GPUScene::shrinkCollisionObjectsArray() {
    auto** newCollisionObjects = new CollisionObject*[currentLength - 128];
    for (int i = 0; i < numCollisionObjects; ++i) {
        newCollisionObjects[i] = collisionObjects[i];
    }
    delete[] collisionObjects;
    collisionObjects = newCollisionObjects;
    currentLength -= 128;
}

#endif