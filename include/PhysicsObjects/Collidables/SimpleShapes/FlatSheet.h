//
// Created by James Miller on 10/9/2024.
//

#ifndef FLATSHEET_H
#define FLATSHEET_H

#include "../CollisionObject.h"

class FlatSheet : public CollisionObject
{
public:
    // Constructors
    FlatSheet();
    explicit FlatSheet(float sideLength);
    explicit FlatSheet(float height, float width);
    explicit FlatSheet(float height, float width, Vector3 position, float mass);

    // Destructor
    ~FlatSheet() override;

    // Variables
    float height;
    float width;

    // Override Functions
    bool isColliding(CollisionObject* other) override;

    bool isTouching(CollisionObject* other) override;

    Vector3 getClosestPoint(const Vector3& point) override;
};

#endif //FLATSHEET_H
