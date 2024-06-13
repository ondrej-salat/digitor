#include "kernel.h"


__global__ void
feedKernel(double *pN, const double *b, const double *w, double *r, double *rR, int rows, int cols) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < rows) {
        double sum = 0;
        for (int i = 0; i < cols; ++i) {
            sum += pN[i] * w[id * cols + i];
        }
        rR[id] = sum + b[id];
        r[id] = (1.0 / (1.0 + exp(-(double) rR[id])));
    }
}

void kernel::doFeedForward(Network &network) {
    for (int i = 1; i < network.network_size; ++i) {
        Layer layer = network.layer[i];
        double *d_pN, *d_b, *d_w, *d_r, *d_rR;

        cudaMalloc(&d_r, sizeof(double) * layer.neurons);
        cudaMalloc(&d_rR, sizeof(double) * layer.neurons);
        cudaMalloc(&d_pN, sizeof(double) * layer.prevNeurons);
        cudaMalloc(&d_b, sizeof(double) * layer.neurons);
        cudaMalloc(&d_w, sizeof(double) * layer.neurons * layer.prevNeurons);
        cudaMemcpy(d_pN, network.layer[i - 1].neuron, sizeof(double) * layer.prevNeurons, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, layer.bias, sizeof(double) * layer.neurons, cudaMemcpyHostToDevice);
        cudaMemcpy(d_w, layer.weight, sizeof(double) * layer.neurons * layer.prevNeurons, cudaMemcpyHostToDevice);
        int BLOCK_SIZE = 32;
        int GRID_SIZE = (layer.neurons * layer.prevNeurons + BLOCK_SIZE - 1) / BLOCK_SIZE;

        feedKernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_pN, d_b, d_w, d_r, d_rR, layer.neurons, layer.prevNeurons);
        cudaDeviceSynchronize();

        cudaMemcpy(layer.neuron, d_r, sizeof(double) * layer.neurons, cudaMemcpyDeviceToHost);
        cudaMemcpy(layer.rawNeuron, d_rR, sizeof(double) * layer.neurons, cudaMemcpyDeviceToHost);
    }

}
