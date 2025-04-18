//
// Created by James Miller on 3/29/2025.
//

#pragma once

#include "../CrossPlatformMacros.h"
#include "../GeneralTypes.h"
#include "MObject.h"
#include "MBounds.h"
#include "PhysicsSim/BasePhysicsFunctions.h"
#include <cmath>
#include <vector>

#include <vector>
#include "MObject.h"

    namespace MillerPhysics {

        class MScene {
        public:
            MScene();
            virtual ~MScene();

            void addObject(MObject* object);
            void checkCollisions() const;
            virtual void BeginPlay();
            virtual void EventTick(float secondsElapsed);
            virtual void EndPlay(const EndPlayEvent& reason);

        private:
            std::vector<MObject*> m_objects;

            friend class MObject; // Allow MObject to access private members
        };

    } // namespace MillerPhysics
