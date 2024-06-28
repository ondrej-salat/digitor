#include <curand_kernel.h>
#include "kernel.h"

__global__ void
feedKernel(const double *pN, const double *b, const double *w, double *r, double *rR, int rows, int cols,
           int fn) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= rows) return;

    double sum = 0;
    for (int i = 0; i < cols; ++i) {
        sum += pN[i] * w[id * cols + i];
    }
    rR[id] = sum + b[id];
    if (fn == 0) {
        double sig;
        if (rR[id] >= 30) sig = 1;
        else if (rR[id] <= -30) sig = 0;
        else sig = (1.0 / (1.0 + exp(-(double) rR[id])));
        r[id] = sig;
    } else if (fn == 1) r[id] = rR[id] > 0 ? rR[id] : 0;
}

__global__ void initRandomKernel(double *b, double *w, int rows, int cols, int seed, bool last) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id >= rows) return;
    curandState state;
    curand_init(seed, id, 0, &state);
    for (int i = 0; i < cols; ++i) {
        w[id * cols + i] = curand_uniform(&state) * 2.0 - 1.0;
    }
    if (!last) b[id] = curand_uniform(&state) * 0.2 - 0.1;
}


__global__ void
doBackpropagationKernel(const double *neuron, const double *rawNeuron, const double *source,
                        const double *rawSource, double *weight, const double *next_weight, double *bias,
                        const double *result,
                        double *deltaError, const double *deltaErrorNext, bool last, int rows, int cols, int fn,
                        double learningRate, int nextNeurons) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (last) {
        if (id >= rows) return;
        deltaError[id] = 2 * (neuron[id] - result[id]);
        for (int i = 0; i < cols; ++i) {
            double localError = deltaError[id];
            if (fn == 0) {
                double sig;
                if (rawSource[i] >= 30) sig = 1;
                else if (rawSource[i] <= -30) sig = 0;
                else sig = (1.0 / (1.0 + exp(-(double) rawSource[i])));
                localError *= sig * (1.0 - sig);
            } else if (fn == 1) {
                localError *= (rawSource[i] >= 0.0 ? 1.0 : 0.0);
            }
            weight[id * cols + i] -= learningRate * localError * source[i];
            //bias[i] -= learningRate * localError;
        }
    } else {
        if (id >= cols) return;
        double sum = 0;
        for (int j = 0; j < nextNeurons; ++j) {
            double activatedDer = 0;
            if (fn == 0) {
                double sig;
                if (rawSource[id] >= 30) sig = 1;
                else if (rawSource[id] <= -30) sig = 0;
                else sig = (1.0 / (1.0 + exp(-(double) rawSource[id])));
                activatedDer = sig * (1 - sig);
            } else if (fn == 1) {
                activatedDer = (rawNeuron[id] >= 0 ? 1 : 0);
            }
            sum += deltaErrorNext[j] * activatedDer * next_weight[j * cols + id];
        }
        deltaError[id] = sum;
        double localError = deltaError[id];
        if (fn == 0) {
            double sig;
            if (rawSource[id] >= 30) sig = 1;
            else if (rawSource[id] <= -30) sig = 0;
            else sig = (1.0 / (1.0 + exp(-(double) rawSource[id])));
            localError *= sig * (1.0 - sig);
        } else if (fn == 1) {
            double relu = rawSource[id] > 0 ? rawSource[id] : 0;
            localError *= (relu >= 0 ? 1 : 0);
        }
        bias[id] -= learningRate * localError;
        for (int i = 0; i < rows; ++i) {
            weight[i * cols + id] -= learningRate * localError * source[id];
        }
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
        unsigned int BLOCK_SIZE = 256;
        unsigned int GRID_SIZE = (layer.neurons * layer.prevNeurons + BLOCK_SIZE - 1) / BLOCK_SIZE;

        feedKernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_pN, d_b, d_w, d_r, d_rR, layer.neurons, layer.prevNeurons,
                                              network.activation);
        cudaDeviceSynchronize();

        cudaMemcpy(layer.neuron, d_r, sizeof(double) * layer.neurons, cudaMemcpyDeviceToHost);
        cudaMemcpy(layer.rawNeuron, d_rR, sizeof(double) * layer.neurons, cudaMemcpyDeviceToHost);

        cudaFree(d_r);
        cudaFree(d_rR);
        cudaFree(d_pN);
        cudaFree(d_b);
        cudaFree(d_w);
    }
}


void kernel::initNetwork(Network &network, int seed) {
    for (int i = 1; i < network.network_size; ++i) {
        Layer layer = network.layer[i];
        double *d_b, *d_w;
        bool last = i == (network.network_size - 1);
        cudaMalloc(&d_b, sizeof(double) * layer.neurons);
        cudaMalloc(&d_w, sizeof(double) * layer.neurons * layer.prevNeurons);
        unsigned int BLOCK_SIZE = 256;
        unsigned int GRID_SIZE = (layer.neurons * layer.prevNeurons + BLOCK_SIZE - 1) / BLOCK_SIZE;
        initRandomKernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_b, d_w, layer.prevNeurons, layer.neurons, seed, last);
        cudaDeviceSynchronize();

        cudaMemcpy(layer.bias, d_b, sizeof(double) * layer.neurons, cudaMemcpyDeviceToHost);
        cudaMemcpy(layer.weight, d_w, sizeof(double) * layer.neurons * layer.prevNeurons, cudaMemcpyDeviceToHost);

        cudaFree(d_b);
        cudaFree(d_w);
    }
}

void kernel::doBackpropagation(Network &network, double learningRate, double *result) {
    double *deltaErrorNext;
    for (int i = network.network_size - 1; i > 0; --i) {
        Layer layer = network.layer[i];
        Layer previous = network.layer[i - 1];
        bool last = i == (network.network_size - 1);
        int nextNeurons = !last ? network.layer[i + 1].neurons : 0;
        double *next_weight = !last ? network.layer[i + 1].weight : nullptr;
        double *d_neuron, *d_rawNeuron, *d_source, *d_rawSource, *d_weight, *d_nextWeight, *d_bias, *d_result, *deltaError;
        cudaMalloc(&d_neuron, sizeof(double) * layer.neurons);
        cudaMalloc(&d_rawNeuron, sizeof(double) * layer.neurons);
        cudaMalloc(&d_source, sizeof(double) * layer.prevNeurons);
        cudaMalloc(&d_rawSource, sizeof(double) * layer.prevNeurons);
        cudaMalloc(&d_weight, sizeof(double) * layer.neurons * layer.prevNeurons);
        cudaMalloc(&d_nextWeight,
                   sizeof(double) * (!last ? network.layer[i + 1].neurons * network.layer[i + 1].prevNeurons : 0));
        cudaMalloc(&d_bias, sizeof(double) * layer.neurons);
        cudaMalloc(&d_result, sizeof(double) * layer.neurons);
        cudaMalloc(&deltaError, sizeof(double) * layer.neurons);

        cudaMemcpy(d_neuron, layer.neuron, sizeof(double) * layer.neurons, cudaMemcpyHostToDevice);
        cudaMemcpy(d_rawSource, layer.rawNeuron, sizeof(double) * layer.neurons, cudaMemcpyHostToDevice);
        cudaMemcpy(d_source, previous.neuron, sizeof(double) * layer.prevNeurons, cudaMemcpyHostToDevice);
        cudaMemcpy(d_rawSource, previous.rawNeuron, sizeof(double) * layer.prevNeurons, cudaMemcpyHostToDevice);
        cudaMemcpy(d_weight, layer.weight, sizeof(double) * layer.neurons * layer.prevNeurons, cudaMemcpyHostToDevice);
        if (!last) {
            cudaMemcpy(d_nextWeight, network.layer[i + 1].weight,
                       sizeof(double) * network.layer[i + 1].neurons * network.layer[i + 1].prevNeurons,
                       cudaMemcpyHostToDevice);
        }
        cudaMemcpy(d_bias, layer.bias, sizeof(double) * layer.neurons, cudaMemcpyHostToDevice);
        cudaMemcpy(d_result, result, sizeof(double) * layer.neurons, cudaMemcpyHostToDevice);


        unsigned int BLOCK_SIZE;
        unsigned int GRID_SIZE;
        unsigned int limit = last ? layer.neurons : layer.prevNeurons;
        if (limit > 512) {
            BLOCK_SIZE = 512;
            GRID_SIZE = (limit + BLOCK_SIZE - 1) / BLOCK_SIZE;
        } else {
            BLOCK_SIZE = limit;
            GRID_SIZE = 1;
        }
        doBackpropagationKernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_neuron, d_rawNeuron,
                                                           d_source, d_rawSource, d_weight, d_nextWeight,
                                                           d_bias, d_result, deltaError, deltaErrorNext,
                                                           last, layer.neurons, layer.prevNeurons,
                                                           network.activation, learningRate, nextNeurons);
        cudaDeviceSynchronize();

        cudaMalloc(&deltaErrorNext, sizeof(double) * layer.neurons);
        cudaMemcpy(&deltaErrorNext, deltaError, sizeof(double) * layer.neurons, cudaMemcpyDeviceToDevice);

        cudaMemcpy(layer.weight, d_weight, sizeof(double) * layer.prevNeurons * layer.neurons, cudaMemcpyDeviceToHost);
        cudaMemcpy(layer.bias, d_bias, sizeof(double) * layer.neurons, cudaMemcpyDeviceToHost);

        cudaFree(d_neuron);
        cudaFree(d_rawNeuron);
        cudaFree(d_source);
        cudaFree(d_rawSource);
        cudaFree(d_weight);
        cudaFree(d_nextWeight);
        cudaFree(d_bias);
        cudaFree(d_result);
        cudaFree(deltaError);
    }
    cudaFree(deltaErrorNext);
}

void kernel::doTraining(Network &network, TrainData data, double learningRate) {
    for (int i = 0; i < data.data_size; ++i) {
        for (int j = 0; j < data.image[i].image_size; ++j) {
            network.layer[0].neuron[j] = data.image[i].image[j];
        }
        this->doFeedForward(network);
        auto *result = new double[network.layer[network.network_size - 1].neurons];
        for (int j = 0; j < network.layer[network.network_size - 1].neurons; ++j) {
            result[j] = j == data.image[i].value;
        }
        this->doBackpropagation(network, learningRate, result);
    }
}