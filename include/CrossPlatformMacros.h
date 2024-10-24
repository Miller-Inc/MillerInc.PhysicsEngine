//
// Created by James Miller on 10/23/2024.
//

#ifndef CROSSPLATFORMMACROS_H
#define CROSSPLATFORMMACROS_H

#include <cuda_runtime.h>

#if checkNvidiaGpu()
#define NVIDIA_GPU
#endif

#endif //CROSSPLATFORMMACROS_H
