//
// Created by James Miller on 3/27/2025.
//

#pragma once

namespace MillerPhysics
{

    typedef enum eventType
    {
        EventBeginPlay = 0,
        EventTick = 1,
        EventEndPlay = 2,
        EventPause = 3,
        EventResume = 4,
        EventRestart = 5,
    } EventType;

    typedef enum endPlayType
    {
        UnknownExit = -1,
        NormalExit = 0,
        UserTerminated = 1,
        CXXException = 2,
        UnhandledException = 3,
        FatalError = 4,
        PauseError = 5,
        ResumeError = 6,
    }EndPlayReason;

    typedef struct endPlayEvent
    {
        EndPlayReason reason;
        int errorCode;

        endPlayEvent()
        {
            reason = EndPlayReason::UnknownExit;
            errorCode = -1;
        }

        explicit endPlayEvent(EventType type)
        {
            switch (type)
            {
            case EventBeginPlay:
                reason = EndPlayReason::UnknownExit;
                errorCode = UnknownExit;
                break;
            case EventTick:
                reason = UnknownExit;
                errorCode = UnknownExit;
                break;
            case EventEndPlay:
                reason = EndPlayReason::NormalExit;
                errorCode = 0;
                break;
            case EventType::EventPause:
                reason = EndPlayReason::PauseError;
                errorCode = PauseError;
                break;
            case EventResume:
                reason = ResumeError;
                errorCode = ResumeError;
                break;
            case EventType::EventRestart:
                reason = EndPlayReason::ResumeError;
                errorCode = ResumeError;
                break;
            default:
                reason = EndPlayReason::UnknownExit;
                errorCode = -1;
                break;
            }
        }

        endPlayEvent(EndPlayReason reason, int errorCode)
        {
            this->reason = reason;
            this->errorCode = errorCode;
        }
    }EndPlayEvent;


} // MillerPhysics