#ifndef DIGITOR_NETWORK_H
#define DIGITOR_NETWORK_H

struct Layer {
    int neurons;
    int prevNeurons;
    double *weight;
    double *bias;
    double *neuron;
    double *rawNeuron;

    void allocateMemory() {
        neuron = new double[neurons];
        rawNeuron = new double[neurons];
        bias = new double[neurons];
        weight = new double[neurons * prevNeurons];
    }


    void freeMemory() const {
        delete[] neuron;
        delete[] rawNeuron;
        delete[] bias;
        delete[] weight;
    }
};


struct Network {
    int network_size;
    int activation;
    Layer *layer;

    void allocateMemory() {
        layer = new Layer[network_size];
    }

    void freeMemory() const {
        delete[] layer;
    }
};


struct Layers {
    int layer_size;
    int *layer;

    void allocateMemory() {
        layer = new int[layer_size];
    }

    void freeMemory() const {
        delete[] layer;
    }
};

enum ActivationFn {
    SIGMOID = 0,
    RELU = 1
};

#endif //DIGITOR_NETWORK_H
