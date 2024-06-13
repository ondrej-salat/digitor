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

void NeuralNetworkCUDA::initRandom() const {
    for (int i = 1; i < network.network_size; ++i) {
        for (int j = 0; j < network.layer[i].neurons; ++j) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> dis(-0.1, 0.1);
            network.layer[i].bias[j] = dis(gen);
            for (int k = 0; k < network.layer[i].prevNeurons; ++k) {
                std::random_device rd2;
                std::mt19937 gen2(rd2());
                std::uniform_real_distribution<> dis2(-1.0, 1.0);
                network.layer[i].weight[j * network.layer[i].prevNeurons + k] = dis2(gen2);
            }
        }
    }
}



