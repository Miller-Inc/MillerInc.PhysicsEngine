//
// Created by James Miller on 11/6/2024.
//

#ifndef GPU_MACROS_H
#define GPU_MACROS_H

// Define whether CUDA is available
// #define CUDA_AVAILABLE true

#if CUDA_AVAILABLE
// Include CUDA kernels
#include "CUDA.Physics/kernel.h"

#endif



#endif //GPU_MACROS_H
