//
// Created by James Miller on 11/6/2024.
//

#ifndef GPUSCENE_H
#define GPUSCENE_H

#if CUDA_AVAILABLE

#define INCREMENT 128

#include "../../include/PhysicsEngine/Scenes/Scene.h"
#include "../../include/PhysicsObjects/Collidables/CollisionObject.h"

class GPUScene {
public:
    GPUScene();
    ~GPUScene();

    void addCollisionObject(CollisionObject* object);
    void removeCollisionObject(CollisionObject* object);

    [[nodiscard]] int getNumCollisionObjects() const;

protected:
    CollisionObject* *collisionObjects;
    int numCollisionObjects, currentLength;

    void enlargeCollisionObjectsArray();
    void shrinkCollisionObjectsArray();


public:
    int* removeIndex; // Index of object to remove
};

#endif

#endif //GPUSCENE_H