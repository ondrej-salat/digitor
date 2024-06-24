#ifndef DIGITOR_NEURALNETWORKCUDA_H
#define DIGITOR_NEURALNETWORKCUDA_H

#include <vector>
#include <random>
#include <string>
#include "kernel.h"


class NeuralNetworkCUDA {
public:
    explicit NeuralNetworkCUDA(Layers layers, ActivationFn activationFn);

    //explicit NeuralNetworkCUDA(const std::string &filename);

    std::vector<double> feed(const std::vector<double> &input);

    void train(const TrainData &data, unsigned int iterations, double learningRate);

private:
    void feedForward();

    void initRandom();

    int activationType{};
    std::string filename;
    Network network{};
};


#endif //DIGITOR_NEURALNETWORKCUDA_H
