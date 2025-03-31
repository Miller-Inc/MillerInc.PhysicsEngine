//
// Created by James Miller on 3/27/2025.
//

#pragma once

#include "../CrossPlatformMacros.h"
#include "../GeneralTypes.h"
#include "MBounds.h"
#include "../SimulationTypes/EventTypes.h"

namespace MillerPhysics {

class MScene;

class MObject {
    // Initializers
    public:
    MObject(); // Default constructor
    MObject(const MObject& obj);
    explicit MObject(const MVector& pos);
    MObject(const MVector& pos, const MQuaternion& rot);
    MObject(MVector pos, MQuaternion rot, float mass);

    virtual ~MObject() = default; // Default destructor

    // Fields
    protected:
    friend class MScene;
    // Simple fields
    float m_mass;
    MVector m_position;
    MVector m_init_position;
    MQuaternion m_rotation{};
    MQuaternion m_init_rotation{};
    MVector m_scale{};
    MVector m_init_scale{};
    bool m_simulate_physics = false;
    PhysicsFunction m_physics_function = nullptr;

    bool m_is_running = false;

public:

    // Collision fields
    MBounds* m_bounds = nullptr;

    public:
    virtual void BeginPlay();
    virtual void EventTick(float secondsElapsed);
    virtual void EndPlay(EndPlayEvent reason);
    virtual void Pause();
    virtual void Resume();
    virtual void Restart();
    void SetMass(float mass);
    [[nodiscard]] float GetMass() const;
    virtual void SetupPhysics(PhysicsFunction function);
    void SetPosition(const MVector& position);
    [[nodiscard]] MVector GetPosition() const;
    void SetRotation(const MQuaternion& rotation);
    [[nodiscard]] MQuaternion GetRotation() const;
    void SetScale(const MVector& scale);
    [[nodiscard]] MVector GetScale() const;
    void SetSimulatePhysics(bool simulate);
    [[nodiscard]] bool GetSimulatePhysics() const;
};

} // MillerPhysics

