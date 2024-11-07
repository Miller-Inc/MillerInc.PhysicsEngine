//
// Created by James Miller on 11/6/2024.
//

#ifndef GPUSCENE_H
#define GPUSCENE_H

#if CUDA_AVAILABLE

#include "../../include/PhysicsEngine/Scenes/Scene.h"

class GPUScene {
public:
    static void runSceneOnGPU(Scene* scene);
};

#endif

#endif //GPUSCENE_H