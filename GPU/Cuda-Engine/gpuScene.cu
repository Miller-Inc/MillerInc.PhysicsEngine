#if CUDA_AVAILABLE

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../../include/PhysicsEngine/Scenes/Scene.h"
#include "gpuScene.h"

__device__ int dev_removeIndex;

__global__ void findCollisionObject(CollisionObject* collision_objects[], int numCollisionObjects, CollisionObject* obj) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < numCollisionObjects) {
        bool result = collision_objects[i]->position == obj->position &&
            collision_objects[i]->velocity == obj->velocity &&
            collision_objects[i]->rotation == obj->rotation &&
            collision_objects[i]->angularVelocity == obj->angularVelocity &&
            collision_objects[i]->mass == obj->mass /*&&
            strcmp(collision_objects[i]->name, obj->name) == 0;*/

        ;

        if (result) {
            dev_removeIndex = i;
        }
    }
}

__global__ void getEqual(std::byte *collObjs, int numCollisionObjects, std::byte *data, int colLen) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < numCollisionObjects) {
        bool result = true;
        for (int j = 0; j < colLen; j++) {
            if (collObjs[i * colLen + j] != data[j]) {
                result = false;
                break;
            }
        }

        if (result) {
            dev_removeIndex = i;
        }
    }
}

void GPUScene::removeCollisionObject(const CollisionObject* object)
{
    int removeIndex = -1;

    std::cout << "Removing object: " << object->name << "\n";

    for (int i = 0; i < numCollisionObjects; i++) {
        if (collisionObjects[i].position == object->position &&
            collisionObjects[i].velocity == object->velocity &&
            collisionObjects[i].rotation == object->rotation &&
            collisionObjects[i].angularVelocity == object->angularVelocity &&
            collisionObjects[i].mass == object->mass /*&&
            strcmp(collisionObjects[i].name, object->name) == 0*/) {
            removeIndex = i;
            break;
        }
    }

    std::cout << "Remove index: " << removeIndex << "\n";

    if (removeIndex == -1) {
        std::cout << "Object not found\n";
        return;
    }

    for (int i = removeIndex; i < numCollisionObjects - 1; i++) {
        collisionObjects[i] = collisionObjects[i + 1];
    }

    numCollisionObjects--;
}

GPUScene::~GPUScene() {
    /*for (int i = 0; i < numCollisionObjects; ++i) {
        delete collisionObjects[i];
    }*/
    free(collisionObjects);
    delete[] collisionObjects;
}

GPUScene::GPUScene() {
    currentLength = 128;
    numCollisionObjects = 0;
    // collisionObjects[currentLength];
    removeIndex = new int;
    *removeIndex = -1;
}

void GPUScene::addCollisionObject(CollisionObject* object) {
    if (numCollisionObjects == currentLength) {
        enlargeCollisionObjectsArray();
    }
    collisionObjects[numCollisionObjects] = *object;
    numCollisionObjects++;
}

int GPUScene::getNumCollisionObjects() const {
    return this->numCollisionObjects;
}

void GPUScene::enlargeCollisionObjectsArray() {
    const int newLength = currentLength + 128;
    auto* newCollisionObjects = new CollisionObject[newLength];
    for (int i = 0; i < numCollisionObjects; ++i) {
        newCollisionObjects[i] = collisionObjects[i];
    }
    delete[] collisionObjects;

    memccpy(collisionObjects, newCollisionObjects, 0, currentLength);

    // collisionObjects = newCollisionObjects;
    currentLength += 128;
}

void GPUScene::shrinkCollisionObjectsArray() {
    const int newLength = currentLength - 128;
    auto* newCollisionObjects = new CollisionObject[newLength];
    for (int i = 0; i < numCollisionObjects; ++i) {
        newCollisionObjects[i] = collisionObjects[i];
    }
    delete[] collisionObjects;

    memccpy(collisionObjects, newCollisionObjects, 0, currentLength);

    // collisionObjects = newCollisionObjects;
    currentLength -= 128;
}

#endif