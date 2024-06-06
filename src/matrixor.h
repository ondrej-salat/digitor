#ifndef MATRIXOR_MATRIXOR_H
#define MATRIXOR_MATRIXOR_H

#include <iostream>
#include <vector>

class matrixor {
public:
    explicit matrixor();

    void feedForwardCalculation(std::vector<std::vector<double>> &n, std::vector<std::vector<double>> &rN,
                                std::vector<std::vector<std::vector<double>>> &w,
                                std::vector<std::vector<double>> &b);


    void multiply(matrixor &other);

private:
    int init;
};

#endif //MATRIXOR_MATRIXOR_H
