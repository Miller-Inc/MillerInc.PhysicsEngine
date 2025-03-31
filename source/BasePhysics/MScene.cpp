//
// Created by James Miller on 3/29/2025.
//

#include "../../include/BasePhysics/MScene.h"
#include <iostream>

namespace MillerPhysics {

    MScene::MScene() = default;

    MScene::~MScene() = default;

    void MScene::addObject(MObject* object) {
        m_objects.push_back(object);
    }

    void MScene::BeginPlay()
    {
        for (auto object : m_objects)
        {
            object->BeginPlay();
        }
    }

    void MScene::EventTick(float secondsElapsed)
    {
        for (auto object : m_objects)
        {
            object->EventTick(secondsElapsed);
        }

        checkCollisions();
    }

    void MScene::EndPlay(const EndPlayEvent& reason)
    {
        for (const auto object : m_objects)
        {
            object->EndPlay(reason);
        }
    }



#if !CUDA_AVAILABLE
    void MScene::checkCollisions() {
        for (size_t i = 0; i < m_objects.size(); ++i) {
            for (size_t j = i + 1; j < m_objects.size(); ++j) {
                if (m_objects[i]->m_bounds->intersects(*m_objects[j]->m_bounds)) {
                    std::cout << "Collision detected between objects " << i << " and " << j << "\n";
                }
            }
        }
    }
#endif
} // namespace MillerPhysics