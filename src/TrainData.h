#ifndef DIGITOR_TRAINDATA_H
#define DIGITOR_TRAINDATA_H

#include <vector>
#include "iostream"

struct Image {
    double *image;
    unsigned int image_size;
    unsigned int value;

    void allocateMemory() {
        image = new double[image_size];
    }

    void freeMemory() const {
        delete[] image;
    }
};

struct TrainData {
    Image *image;
    unsigned int data_size;
    unsigned int set;

    void allocateMemory() {
        image = (Image *) malloc(sizeof(Image) * data_size);
    }

    void freeMemory() const {
        free(image);
    }
};


#endif //DIGITOR_TRAINDATA_H
