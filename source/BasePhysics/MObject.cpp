//
// Created by James Miller on 3/27/2025.
//

#include "../../include/BasePhysics/MObject.h"

namespace MillerPhysics {
    MObject::MObject()
    {
        m_position = MVector(0, 0, 0);
        m_init_position = MVector(0, 0, 0);
        m_rotation = MQuaternion(0, 0, 0, 1);
        m_init_rotation = MQuaternion(0, 0, 0, 1);
        m_scale = MVector(1, 1, 1);
        m_init_scale = MVector(1, 1, 1);
        m_bounds = new MBounds(m_position - (m_scale / 2), m_position + (m_scale / 2));
        m_mass = 1.0f;
    }

    MObject::MObject(const MVector& pos)
    {
        m_position = pos;
        m_init_position = m_position;
        m_rotation = MQuaternion(0, 0, 0, 1);
        m_init_rotation = m_rotation;
        m_scale = MVector(1, 1, 1);
        m_init_scale = m_scale;
        m_bounds = new MBounds(m_position - (m_scale / 2), m_position + (m_scale / 2));
        m_mass = 1.0f;
    }

    MObject::MObject(const MObject& obj)
    {
        m_position = obj.m_position;
        m_init_position = m_position;
        m_rotation = obj.m_rotation;
        m_init_rotation = m_rotation;
        m_scale = obj.m_scale;
        m_init_scale = m_scale;
        m_bounds = new MBounds(m_position - (m_scale / 2), m_position + (m_scale / 2));
        m_mass = obj.m_mass;
    }

    MObject::MObject(const MVector& pos, const MQuaternion& rot)
    {
        m_position = pos;
        m_init_position = m_position;
        m_rotation = rot;
        m_init_rotation = m_rotation;
        m_scale = MVector(1, 1, 1);
        m_init_scale = m_scale;
        m_bounds = new MBounds(m_position - (m_scale / 2), m_position + (m_scale / 2));
        m_mass = 1.0f;
    }

    MObject::MObject(MVector pos, MQuaternion rot, float mass)
    {
        m_position = pos;
        m_init_position = m_position;
        m_rotation = rot;
        m_init_rotation = m_rotation;
        m_scale = MVector(1, 1, 1);
        m_init_scale = m_scale;
        m_bounds = new MBounds(m_position - (m_scale / 2), m_position + (m_scale / 2));
        m_mass = mass;
    }

    void MObject::BeginPlay()
    {
        m_position = m_init_position;
        m_rotation = m_init_rotation;
        m_scale = m_init_scale;
        m_is_running = true;
    }

    void MObject::EventTick(const float secondsElapsed)
    {
        if (m_is_running)
        {
            if (m_physics_function != nullptr)
            {
                m_physics_function(*this, secondsElapsed);
            }
        }
    }

    void MObject::EndPlay(EndPlayEvent reason)
    {
        m_is_running = false;
        #if CUDA_AVAILABLE
        cudaFree(m_bounds->gpu_center);
        cudaFree(m_bounds->gpu_max);
        cudaFree(m_bounds->gpu_min);
        #endif
    }

    void MObject::Pause()
    {
        m_is_running = false;
    }

    void MObject::Resume()
    {
        m_is_running = true;
    }

    void MObject::Restart()
    {
        m_position = m_init_position;
        m_rotation = m_init_rotation;
        m_scale = m_init_scale;
        m_is_running = true;
    }

    void MObject::SetMass(const float mass)
    {
        m_mass = mass;
    }

    float MObject::GetMass() const
    {
        return m_mass;
    }

    void MObject::SetupPhysics(const PhysicsFunction function)
    {
        m_physics_function = function;
    }

    void MObject::SetPosition(const MVector& position)
    {
        m_position = position;

        if (!m_is_running)
        {
            m_init_position = position;
        }
    }

    MVector MObject::GetPosition() const
    {
        return m_position;
    }

    MQuaternion MObject::GetRotation() const
    {
        return m_rotation;
    }

    MVector MObject::GetScale() const
    {
        return m_scale;
    }

    bool MObject::GetSimulatePhysics() const
    {
        return m_simulate_physics;
    }

    void MObject::SetRotation(const MQuaternion& rotation)
    {
        m_rotation = rotation;

        if (!m_is_running)
        {
            m_init_rotation = rotation;
        }
    }

    void MObject::SetScale(const MVector& scale)
    {
        m_scale = scale;

        if (!m_is_running)
        {
            m_init_scale = scale;
        }
    }

    void MObject::SetSimulatePhysics(bool simulate)
    {
        m_simulate_physics = simulate;
    }
} // MillerPhysics