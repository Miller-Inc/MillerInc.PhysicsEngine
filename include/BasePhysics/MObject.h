//
// Created by James Miller on 3/27/2025.
//

#pragma once

#include "../CrossPlatformMacros.h"
#include "../GeneralTypes.h"
#include "MBounds.h"
#include "../SimulationTypes/EventTypes.h"

namespace MillerPhysics {

class MObject {
    // Initializers
    public:
    MObject(); // Default constructor

    virtual ~MObject() = default; // Default destructor

    // Fields
    protected:
    // Simple fields
    MVector m_position;
    MVector m_init_position;
    MQuaternion m_rotation{};
    MQuaternion m_init_rotation{};
    MVector m_scale{};
    MVector m_init_scale{};
    bool m_simulate_physics = false;
    PhysicsFunction m_physics_function = nullptr;

    bool m_is_running = false;

    // Collision fields
    MBounds* m_bounds = nullptr;

    public:
    virtual void BeginPlay();
    virtual void EventTick(float secondsElapsed);
    virtual void EndPlay(EndPlayEvent reason);
    virtual void Pause();
    virtual void Resume();
    virtual void Restart();
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

