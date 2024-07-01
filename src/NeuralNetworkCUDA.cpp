#include "NeuralNetworkCUDA.h"

using json = nlohmann::json;

NeuralNetworkCUDA::NeuralNetworkCUDA(Layers &layers, ActivationFn activationFn) {
    layers = layers;
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
    network.activation = activationType;
    initRandom();
    initJsonFile();
}

NeuralNetworkCUDA::NeuralNetworkCUDA(const std::string &filename) {
    this->filename = filename;
    json data = readJsonFile();
    activationType = data["activation"];
    network.network_size = data["networkSize"];
    network.allocateMemory();
    layers.layer_size = network.network_size;
    layers.allocateMemory();
    for (int i = 0; i < network.network_size; ++i) {
        network.layer[i].neurons = data["layer"][i]["neurons"];
        layers.layer[i] = network.layer[i].neurons;
        network.layer[i].prevNeurons = data["layer"][i]["prevNeurons"];
        network.layer[i].allocateMemory();
        for (int j = 0; j < network.layer[i].neurons; ++j) {
            network.layer[i].bias[j] = data["layer"][i]["bias"][j];
            for (int k = 0; k < network.layer[i].prevNeurons; ++k) {
                network.layer[i].weight[j * network.layer[i].prevNeurons + k] = data["layer"][i]["weight"][
                        j * network.layer[i].prevNeurons + k];
            }
        }
    }

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

void NeuralNetworkCUDA::train(TrainData &data, unsigned int iterations, double learningRate) {
    double progress;
    for (int i = 0; i < iterations; ++i) {
        progress = (double) i * 100 / iterations;
        std::cout << "\r" << std::fixed << std::setprecision(2) << progress;
        kernel k;
        k.doTraining(network, data, learningRate);
    }
    std::cout << "\n";
}

nlohmann::json NeuralNetworkCUDA::readJsonFile() {
    std::ifstream jFile(filename);
    if (!jFile.is_open()) {
        std::cerr << "Failed to open the JSON file." << std::endl;
        exit(EXIT_FAILURE);
    }
    std::string jsonString((std::istreambuf_iterator<char>(jFile)), std::istreambuf_iterator<char>());
    jFile.close();
    return json::parse(jsonString);
}

void NeuralNetworkCUDA::initJsonFile() {
    std::string file;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1.0);
    file += "network_s" + std::to_string(network.network_size) + "_i" +
            std::to_string((int) round(dis(gen) * 10000)) +
            ".json";
    std::ofstream output_file(file);
    filename = file;
    writeJson();
}

void NeuralNetworkCUDA::writeJson() {
    std::ofstream output_file(filename);
    if (output_file.is_open()) {
        json data;
        data["activation"] = activationType;
        data["networkSize"] = network.network_size;
        for (int i = 0; i < network.network_size; ++i) {
            data["layer"][i]["neurons"] = network.layer[i].neurons;
            data["layer"][i]["prevNeurons"] = network.layer[i].prevNeurons;
            for (int j = 0; j < network.layer[i].neurons; ++j) {
                data["layer"][i]["bias"][j] = network.layer[i].bias[j];
                for (int k = 0; k < network.layer[i].prevNeurons; ++k) {
                    data["layer"][i]["weight"][j * network.layer[i].prevNeurons + k] = network.layer[i].weight[
                            j * network.layer[i].prevNeurons + k];
                }
            }
        }
        std::string stringData = data.dump(2);
        output_file << stringData;
        output_file.close();
        std::cout << "JSON saved to '" << filename << "'" << std::endl;
    } else {
        std::cerr << "Failed to open the output file." << std::endl;
    }
}
