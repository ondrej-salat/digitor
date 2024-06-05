#ifndef MATRIXOR_MATRIXOR_H
#define MATRIXOR_MATRIXOR_H

#include <iostream>
#include <vector>

class matrixor {
public:
    explicit matrixor(std::vector<std::vector<double>> &matrix);

    void copy(matrixor &other);

    void destroy();

    void multiply(matrixor &other);

    void print() const;

    double *data;
    unsigned int rows;
    unsigned int cols;
private:
};

#endif //MATRIXOR_MATRIXOR_H
