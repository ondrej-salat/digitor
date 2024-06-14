#include "NeuralNetworkCUDA.h"

NeuralNetworkCUDA::NeuralNetworkCUDA(Layers layers, ActivationFn activationFn) {
    activationType = activationFn;
    this->network.network_size = layers.layer_size;
    network.allocateMemory();
    network.layer[0].neurons = layers.layer[0];
    network.layer[0].prevNeurons = 0;
    network.layer[0].allocateMemory();
    for (int i = 1; i < layers.layer_size; ++i) {
        network.layer[i].neurons = layers.layer[i];
        network.layer[i].prevNeurons = layers.layer[i - 1];
        network.layer[i].allocateMemory();
    }
    initRandom();
}

std::vector<double> NeuralNetworkCUDA::feed(const std::vector<double> &input) {
    Layer l = network.layer[0];
    if (input.size() != l.neurons) {
        std::cerr << "Wrong input size" << "\n";
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < l.neurons; ++i) {
        l.neuron[i] = input[i];
    }
    feedForward();
    std::vector<double> output;
    for (int i = 0; i < network.layer[network.network_size - 1].neurons; ++i) {
        output.push_back(network.layer[network.network_size - 1].neuron[i]);
    }
    return output;
}

void NeuralNetworkCUDA::feedForward() {
    kernel k = kernel();
    k.doFeedForward(network);
}

void NeuralNetworkCUDA::initRandom() {
    kernel k;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1000);
    k.initNetwork(network, (int) dis(gen));
}

void NeuralNetworkCUDA::train(const TrainData &data, unsigned int iterations, double learningRate) {
    for (int i = 0; i < iterations; ++i) {


    }
}
