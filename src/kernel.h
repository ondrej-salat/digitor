#ifndef DIGITOR_KERNEL_H
#define DIGITOR_KERNEL_H

#include "Network.h"
#include "TrainData.h"


class kernel {
public:
    void doFeedForward(Network &network);

    void initNetwork(Network &network, int seed);

    void doTraining(Network &network, TrainData &data, double learningRate);

    void doBackpropagation(Network &network, double learningRate, double *result);

};

#endif //DIGITOR_KERNEL_H
