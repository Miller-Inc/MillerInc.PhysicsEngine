#include <stdexcept>
#if CUDA_AVAILABLE
#include "../../../include/FieldTypes/BaseTypes/Vector3.h"
#include "../../../include/FieldTypes/BaseTypes/Vector3Math.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void addKernel(const Vector3* a, const Vector3* b, Vector3* result, int n) {
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        result[i].x = a[i].x + b[i].x;
        result[i].y = a[i].y + b[i].y;
        result[i].z = a[i].z + b[i].z;
    }
}

__global__ void multiplyKernel(const Vector3* a, const float* b, Vector3* result, int n) {
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        result[i].x = a[i].x * b[i];
        result[i].y = a[i].y * b[i];
        result[i].z = a[i].z * b[i];
    }
}

__global__ void multiplyVectorKernel(const Vector3* a, const Vector3* b, Vector3* result, int n) {
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        result[i].x = a[i].x * b[i].x;
        result[i].y = a[i].y * b[i].y;
        result[i].z = a[i].z * b[i].z;
    }
}

void Vector3Math::addVectors(const std::vector<Vector3>& a, const std::vector<Vector3>& b, std::vector<Vector3>& result) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must be of the same length");
    }

    unsigned int n = a.size();
    result.resize(n);

    Vector3 *dev_a = nullptr, *dev_b = nullptr, *dev_result = nullptr;
    cudaMalloc((void**)&dev_a, n * sizeof(Vector3));
    cudaMalloc((void**)&dev_b, n * sizeof(Vector3));
    cudaMalloc((void**)&dev_result, n * sizeof(Vector3));

    cudaMemcpy(dev_a, a.data(), n * sizeof(Vector3), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b.data(), n * sizeof(Vector3), cudaMemcpyHostToDevice);

    int blockSize = 256;
    unsigned int numBlocks = (n + blockSize - 1) / blockSize;
    addKernel<<<numBlocks, blockSize>>>(dev_a, dev_b, dev_result, n);

    cudaMemcpy(result.data(), dev_result, n * sizeof(Vector3), cudaMemcpyDeviceToHost);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_result);
}

void Vector3Math::multiplyVectors(const std::vector<Vector3>& a, const std::vector<float>& b, std::vector<Vector3>& result) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors and scalars must be of the same length");
    }

    unsigned int n = a.size();
    result.resize(n);

    Vector3 *dev_a = nullptr, *dev_result = nullptr;
    float *dev_b = nullptr;
    cudaMalloc((void**)&dev_a, n * sizeof(Vector3));
    cudaMalloc((void**)&dev_b, n * sizeof(float));
    cudaMalloc((void**)&dev_result, n * sizeof(Vector3));

    cudaMemcpy(dev_a, a.data(), n * sizeof(Vector3), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    unsigned int numBlocks = (n + blockSize - 1) / blockSize;
    multiplyKernel<<<numBlocks, blockSize>>>(dev_a, dev_b, dev_result, n);

    cudaMemcpy(result.data(), dev_result, n * sizeof(Vector3), cudaMemcpyDeviceToHost);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_result);
}

void Vector3Math::multiplyVectors(const std::vector<Vector3>& a, const std::vector<Vector3>& b, std::vector<Vector3>& result) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must be of the same length");
    }

    unsigned int n = a.size();
    result.resize(n);

    Vector3 *dev_a = nullptr, *dev_b = nullptr, *dev_result = nullptr;
    cudaMalloc((void**)&dev_a, n * sizeof(Vector3));
    cudaMalloc((void**)&dev_b, n * sizeof(Vector3));
    cudaMalloc((void**)&dev_result, n * sizeof(Vector3));

    cudaMemcpy(dev_a, a.data(), n * sizeof(Vector3), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b.data(), n * sizeof(Vector3), cudaMemcpyHostToDevice);

    int blockSize = 256;
    unsigned int numBlocks = (n + blockSize - 1) / blockSize;
    multiplyVectorKernel<<<numBlocks, blockSize>>>(dev_a, dev_b, dev_result, n);

    cudaMemcpy(result.data(), dev_result, n * sizeof(Vector3), cudaMemcpyDeviceToHost);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_result);
}

#endif