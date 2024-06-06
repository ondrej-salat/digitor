#include "matrixor.h"

#include <iostream>
#include <vector>


__global__ void
feedForwardKernel(double *device_neuron, double *device_weight, double *device_bias, double *device_result,
                  double *device_resultA, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < cols && j < rows) {
        double sum = 0.0;
        for (int k = 0; k < cols; ++k) {
            sum += device_weight[j * rows + k] * device_neuron[k];
        }
        device_result[j] = sum + device_bias[i];
        device_resultA[j] = (1.0 / (1.0 + exp(-(double) device_result[j])));
    }
}

matrixor::matrixor() {
    init = 1;
}


void matrixor::feedForwardCalculation(std::vector<std::vector<double>> &n, std::vector<std::vector<double>> &rN,
                                      std::vector<std::vector<std::vector<double>>> &w,
                                      std::vector<std::vector<double>> &b) {
    for (int i = 1; i < n.size(); ++i) {
        double *neuron = (double *) malloc(sizeof(double) * n[i - 1].size());
        double *weights = (double *) malloc(sizeof(double) * w[i - 1].size() * w[i - 1][0].size());
        double *bias = (double *) malloc(sizeof(double) * b[i].size());
        double *result = (double *) malloc(sizeof(double) * n[i].size());
        double *resultA = (double *) malloc(sizeof(double) * n[i].size());
        double *device_n, *device_w, *device_b, *device_result, *device_resultA;
        int rows = w[i - 1][0].size();
        int cols = n[i - 1].size();
        cudaMalloc(&device_n,
                   sizeof(double) * n[i - 1].size());
        cudaMalloc(&device_w, sizeof(double) * w[i - 1].size() * w[i - 1][0].size());
        cudaMalloc(&device_b, sizeof(double) * b[i].size());
        cudaMalloc(&device_result, sizeof(double) * n[i].size());
        cudaMalloc(&device_resultA, sizeof(double) * n[i].size());

        for (int j = 0; j < n[i - 1].size(); ++j) {
            neuron[j] = n[i - 1][j];
        }
        for (int j = 0; j < b[i].size(); ++j) {
            bias[j] = b[i][j];
        }
        for (int j = 0; j < w[i - 1].size(); ++j) {
            for (int k = 0; k < w[i - 1][j].size(); ++k) {
                weights[j * w[i - 1][j].size() + k] = w[i - 1][j][k];
            }
        }
        cudaMemcpy(device_n, neuron, sizeof(double) * n[i - 1].size(), cudaMemcpyHostToDevice);
        cudaMemcpy(device_w, weights, sizeof(double) * w[i - 1].size() * w[i - 1][0].size(), cudaMemcpyHostToDevice);
        cudaMemcpy(device_b, bias, sizeof(double) * b[i].size(), cudaMemcpyHostToDevice);
        cudaMemcpy(device_result, result, sizeof(double) * n[i].size(), cudaMemcpyHostToDevice);
        cudaMemcpy(device_resultA, resultA, sizeof(double) * n[i].size(), cudaMemcpyHostToDevice);

        int BLOCK_SIZE = 32;
        unsigned int grid_rows = (w[i - 1].size() + BLOCK_SIZE - 1) / BLOCK_SIZE;
        unsigned int grid_cols = (n[i - 1].size() + BLOCK_SIZE - 1) / BLOCK_SIZE;
        dim3 dimGrid(grid_cols, grid_rows);
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

        feedForwardKernel<<<dimGrid, dimBlock>>>(device_n, device_w, device_b, device_result, device_resultA, rows,
                                                 cols);
        cudaMemcpy(result, device_result, sizeof(double) * n[i].size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(resultA, device_resultA, sizeof(double) * n[i].size(), cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();
        for (int j = 0; j < n[i].size(); ++j) {
            rN[i][j] = result[j];
            n[i][j] = resultA[j];
        }
    }
}



