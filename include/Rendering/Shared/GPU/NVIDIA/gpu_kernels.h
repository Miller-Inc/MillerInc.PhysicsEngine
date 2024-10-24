//
// Created by James Miller on 10/23/2024.
//

#ifndef GPU_KERNELS_H
#define GPU_KERNELS_H

void launchUpdatePositions(float* positions, float* velocities, int numObjects, float timestep);

#endif // GPU_KERNELS_H
