#include<bits/stdc++.h>
#include "NeuralNetworkCUDA.h"

using namespace std;

NeuralNetworkCUDA *n = nullptr;
bool train = false; // -t

void signalHandler(int signal) {
    if (signal == SIGINT) {
        std::cout << "\nCtrl+C detected. Saving progress and exiting...\n";
        if (train) {
            //n->saveProgress();
        }
        delete n;
        exit(130);
    }
}


static void printVector(vector<double> v) {
    cout << "[";
    for (int i = 0; i < v.size(); ++i) {
        if (i == 0) cout << v[i];
        else cout << ", " << v[i];
    }
    cout << "]" << endl;
}


static vector<unsigned int> charArrToVector(const char *input) {
    std::vector<unsigned int> values;
    stringstream ss(input);
    string word;
    while (!ss.eof()) {
        getline(ss, word, ',');
        values.push_back(stoi(word));
    }
    return values;
}

static Layers inputToLayers(const char *input) {
    std::vector<unsigned int> values;
    stringstream ss(input);
    string word;
    while (!ss.eof()) {
        getline(ss, word, ',');
        values.push_back(stoi(word));
    }
    Layers layer;
    layer.layer_size = values.size();
    layer.allocateMemory();
    for (int i = 0; i < values.size(); ++i) {
        layer.layer[i] = values[i];
    }
    return layer;
}

/*int main() {
    Layers l;
    l.layer_size = 3;
    l.allocateMemory();
    l.layer[0] = 100;
    l.layer[1] = 10;
    l.layer[2] = 10;
    //NeuralNetworkCUDA n = NeuralNetworkCUDA(l, SIGMOID);
    NeuralNetworkCUDA n = NeuralNetworkCUDA("network_s3_i5939.json");
    std::cout << "init" << "\n";
    std::vector<double> intput = {10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
                                  10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
                                  10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
                                  10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
                                  10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10};
    printVector(n.feed(intput));
    TrainData data{};
    data.data_size = 1;
    data.set = 1;
    data.allocateMemory();
    data.image[0].value = 1;
    data.image[0].image_size = 100;
    data.image[0].allocateMemory();
    for (int i = 0; i < data.image[0].image_size; ++i) {
        data.image[0].image[i] = 10;
    }
    //n.train(data, 100, 0.01);
    printVector(n.feed(intput));
}*/

int main(int argc, char *argv[]) {

    std::signal(SIGINT, signalHandler);

    if (argc < 2) {
        cerr << "Invalid output: Must contain at least one argument with the name of neural network file." << endl;
    }

    bool newFile = false; // -n
    const char *filename;
    ActivationFn activationFn;
    Layers neuronsPerLayer{};
    unsigned int iterations;
    unsigned int data_size;
    unsigned int batches;
    double learningRate;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-t") == 0) {
            train = true;
        } else if (strcmp(argv[i], "-n") == 0) {
            newFile = true;
        }
    }

    if (newFile && train) {
        if (argc < 8) {
            cerr << "Missing arguments" << endl
                 << "Expected: `./digitor -t -n <neurons_per_layer> <iterations> <learning_rate> <activation_function> <batches> <batch_size>`"
                 << endl;
            exit(EXIT_FAILURE);
        }
        neuronsPerLayer = inputToLayers(argv[3]);
        iterations = stoi(argv[4]);
        learningRate = stod(argv[5]);
        if (strcmp(argv[6], "sigmoid") == 0) {
            activationFn = SIGMOID;
        } else if (strcmp(argv[6], "relu") == 0) {
            activationFn = RELU;
        } else {
            cerr << "Unsupported activation function \n";
            exit(EXIT_FAILURE);
        }
        batches = stoi(argv[7]);
        data_size = stoi(argv[8]);
        n = new NeuralNetworkCUDA(neuronsPerLayer, activationFn);
        TrainData trainData{};
        trainData.data_size = data_size;
        trainData.set = batches;
        trainData.allocateMemory();
        for (int j = 0; j < data_size; ++j) {
            Image image;
            image.image_size = neuronsPerLayer.layer[0];
            image.allocateMemory();
            for (int k = 0; k < image.image_size; ++k) {
                cin >> image.image[k];
            }
            unsigned int target;
            cin >> target;
            image.value = target;
            trainData.image[j] = image;
        }
        n->train(trainData, iterations, learningRate);
    } else if (train) {
        if (argc < 6) {
            cerr << "Missing arguments" << endl
                 << "Expected: `./digitor -t <filename> <iterations> <learning_rate> <batches> <batch_size>`"
                 << endl;
            exit(EXIT_FAILURE);
        }
        filename = argv[2];
        iterations = stoi(argv[3]);
        learningRate = stod(argv[4]);
        batches = stoi(argv[5]);
        data_size = stoi(argv[6]);
        n = new NeuralNetworkCUDA(filename);
        TrainData trainData{};
        trainData.data_size = data_size;
        trainData.set = batches;
        trainData.allocateMemory();
        for (int j = 0; j < data_size; ++j) {
            Image image;
            image.image_size = neuronsPerLayer.layer[0];
            image.allocateMemory();
            for (int k = 0; k < image.image_size; ++k) {
                cin >> image.image[k];
            }
            unsigned int target;
            cin >> target;
            image.value = target;
            trainData.image[j] = image;
        }
        n->train(trainData, iterations, learningRate);
    } else if (newFile) {
        if (argc < 4) {
            cerr << "Missing arguments" << endl
                 << "Expected: `./digitor -n <neurons_per_layer> <activation_function>`"
                 << endl;
            exit(EXIT_FAILURE);
        }
        neuronsPerLayer = inputToLayers(argv[2]);
        if (strcmp(argv[6], "sigmoid") == 0) {
            activationFn = SIGMOID;
        } else if (strcmp(argv[6], "relu") == 0) {
            activationFn = RELU;
        } else {
            cerr << "Unsupported activation function \n";
            exit(EXIT_FAILURE);
        }
        n = new NeuralNetworkCUDA(neuronsPerLayer, activationFn);
        while (true) {
            vector<double> input(neuronsPerLayer.layer[0]);
            for (double &i: input) {
                cin >> i;
            }
            printVector(n->feed(input));
        }
    } else {
        if (argc < 2) {
            cerr << "Missing arguments" << endl
                 << "Expected: `./digitor <filename>`"
                 << endl;
            exit(EXIT_FAILURE);
        }
        filename = argv[1];
        n = new NeuralNetworkCUDA(filename);
        while (true) {
            vector<double> input(n->layers.layer[0]);
            for (double &i: input) {
                cin >> i;
            }
            vector<double> output = n->feed(input);
            unsigned int estimation = max_element(output.begin(), output.end()) - output.begin();
            cout << estimation << endl;
        }
    }
    return 0;
}