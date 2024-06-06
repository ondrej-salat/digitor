#include "matrixor.h"

#include <iostream>


__global__ void matrixMultiplicationKernel(double *a, double *b, double *c, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    double sum = 0;
    if (col < k && row < m) {
        for (int i = 0; i < n; i++) {
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
}

matrixor::matrixor(std::vector<std::vector<double>> &matrix) {
    rows = matrix.size();
    cols = matrix[0].size();
    data = (double *) malloc(sizeof(double) * rows * cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            data[i * cols + j] = matrix[i][j];
        }
    }
}

void matrixor::destroy() {
    free(data);
    rows = 0;
    cols = 0;
}

void matrixor::copy(matrixor &other) {
    data = other.data;
    rows = other.rows;
    cols = other.cols;
}

void matrixor::multiply(matrixor &other) {
    int m = rows;
    int n = cols;
    int k = other.rows;

    double *device_A, *device_B, *device_result;
    cudaMalloc(&device_A, sizeof(double) * m * n);
    cudaMalloc(&device_B, sizeof(double) * n * k);
    cudaMalloc(&device_result, sizeof(double) * m * k);

    cudaMemcpy(device_A, data, sizeof(double) * m * n, cudaMemcpyHostToDevice);
    cudaMemcpy(device_B, other.data, sizeof(double) * n * k, cudaMemcpyHostToDevice);

    int BLOCK_SIZE = 16;
    unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    matrixMultiplicationKernel<<<dimGrid, dimBlock>>>(device_A, device_B, device_result, m, n, k);
    cudaMemcpy(data, device_result, sizeof(double) * m * k, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaFree(device_A);
    cudaFree(device_B);
    cudaFree(device_result);
    rows = m;
    cols = k;
}


void matrixor::print() const {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << data[i * cols + j] << " ";
        }
        std::cout << "\n";
    }
}