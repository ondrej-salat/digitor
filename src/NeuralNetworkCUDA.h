//
// Created by ondrej on 12/06/24.
//

#ifndef DIGITOR_NEURALNETWORKCUDA_H
#define DIGITOR_NEURALNETWORKCUDA_H

#include <vector>
#include <random>
#include <string>
#include "TrainData.h"
#include "kernel.h"


class NeuralNetworkCUDA {
public:
    explicit NeuralNetworkCUDA(Layers layers, ActivationFn activationFn);

    //explicit NeuralNetworkCUDA(const std::string &filename);

    std::vector<double> feed(const std::vector<double> &input);

    void train(const std::vector<std::vector<TrainData>> &data, unsigned int iterations, double learningRate);

private:
    void feedForward();

    void backPropagate(std::vector<double> target,
                       std::vector<std::vector<double>> relativeDeltaErrors,
                       double learningRate,
                       std::vector<std::vector<std::vector<double>>> &newWeights,
                       std::vector<std::vector<double>> &newBiases
    );

    void initRandom() const;

    void setActivationType(int v);

    static double ReLU(double v);

    static double sigmoid(double v);

    [[nodiscard]] double activationFn(double v) const;

    [[nodiscard]] double activationFnDerivative(double v) const;

    int activationType{};
    std::string filename;
    Network network{};

    double calculateCost(unsigned int targetValue);

    static double sigmoidDerivative(double v);

    static double ReLUDerivative(double v);
};


#endif //DIGITOR_NEURALNETWORKCUDA_H
