//
// Created by James Miller on 10/8/2024.
//
#include "../../../include/PhysicsEngine/Scenes/Scene.h"
#include <chrono>
#include <ctime>
#include <iostream>

void Scene::main()
{
    sceneRunning = true;
    while (sceneRunning)
    {
        step(getDeltaTime());
    }
}

void Scene::main(float time)
{
    float currentTime = 0.0f, deltaTime = 0.0f;
    sceneRunning = true;
    while (sceneRunning && currentTime < time)
    {
        deltaTime = getDeltaTime();
        step(deltaTime);
        currentTime += deltaTime;
    }
}

void Scene::main(float time, bool print)
{
    float currentTime = 0.0f, deltaTime = getDeltaTime();
    sceneRunning = true;
    while (sceneRunning && currentTime < time)
    {
        deltaTime = getDeltaTime();
        step(deltaTime);
        currentTime += deltaTime;
        if (print)
        {
            std::cout << "Time: " << currentTime << " Delta Time: " << deltaTime <<"\n";
            for (auto& object : sceneObjects)
            {
                std::cout << object->toString() << "\n";
            }
        }
    }
}

std::vector<CollisionObject*> Scene::getCollisions(CollisionObject* object)
{
    std::vector<CollisionObject*> collisions;

    for (auto collisionObject : collisionObjects)
    {
        if ((*object == *collisionObject) == false)
        {
            if (object->isColliding(collisionObject))
            {
                collisions.push_back(collisionObject);
            }
        }
    }

    return collisions;
}

void Scene::step(float timestep)
{
    for (auto* object : sceneObjects)
    {
        object->step(timestep);
    }
}

void Scene::AddObject(BaseObject* object)
{
    sceneObjects.push_back(object);
    if (object->isCollidable())
    {
        collisionObjects.push_back(dynamic_cast<CollisionObject*>(object));
    }
}

float Scene::getDeltaTime() {
    auto currentTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> deltaTime = std::chrono::duration_cast<std::chrono::duration<float>>(currentTime - previousTime);
    previousTime = currentTime;
    return deltaTime.count();
}

void Scene::stopSimulation()
{
    sceneRunning = false;
}