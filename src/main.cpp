#include "NeuralNetwork.h"
#include "iostream"
#include "TrainData.h"
#include<bits/stdc++.h>

using namespace std;

volatile std::sig_atomic_t g_signalStatus = 0;

NeuralNetwork *n = nullptr;


void signalHandler(int signal) {
    if (signal == SIGINT) {
        std::cout << "\nCtrl+C detected. Saving progress and exiting...\n";
        g_signalStatus = 1;
        n->saveProgress();
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


int main(int argc, char *argv[]) {

    std::signal(SIGINT, signalHandler);

    if (argc < 1) {
        cerr << "Invalid output: Must contain at least one argument with the name of neural network file." << endl;
    }

    bool train = false; // -t
    bool newFile = false; // -n
    const char *filename;
    const char *activationFn;
    vector<unsigned int> neuronsPerLayer;
    unsigned int iterations;
    unsigned int data_size;
    double learningRate;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-t") == 0) {
            train = true;
        } else if (strcmp(argv[i], "-n") == 0) {
            newFile = true;
        }
    }

    if (newFile && train) {
        neuronsPerLayer = charArrToVector(argv[3]);
        iterations = stoi(argv[4]);
        learningRate = stod(argv[5]);
        activationFn = argv[6];
        data_size = stoi(argv[7]);
        n = new NeuralNetwork(neuronsPerLayer, activationFn);
        vector<TrainData> trainData;
        for (int i = 0; i < data_size; ++i) {
            vector<double> input(neuronsPerLayer[0]);
            for (int j = 0; j < neuronsPerLayer[0]; ++j) {
                cin >> input[j];
            }
            unsigned int target;
            cin >> target;
            trainData.push_back({input, target});
        }
        n->train(trainData, iterations, learningRate);
    } else if (train) {
        filename = argv[2];
        iterations = stoi(argv[3]);
        learningRate = stod(argv[4]);
        data_size = stoi(argv[5]);
        n = new NeuralNetwork(filename);
        vector<TrainData> trainData;
        for (int i = 0; i < data_size; ++i) {
            vector<double> input(n->layers[0]);
            for (int j = 0; j < n->layers[0]; ++j) {
                cin >> input[j];
            }
            int target;
            cin >> target;
            trainData.push_back({input, static_cast<unsigned int>(target)});
        }
        n->train(trainData, iterations, learningRate);
    } else if (newFile) {
        neuronsPerLayer = charArrToVector(argv[2]);
        activationFn = argv[3];
        n = new NeuralNetwork(neuronsPerLayer, activationFn);
        while (true) {
            vector<double> input(neuronsPerLayer[0]);
            for (int i = 0; i < neuronsPerLayer[0]; ++i) {
                cin >> input[i];
            }
            printVector(n->feed(input));
        }
    } else {
        filename = argv[1];
        n = new NeuralNetwork(filename);
        while (true) {
            vector<double> input(n->layers[0]);
            for (int i = 0; i < n->layers[0]; ++i) {
                cin >> input[i];
            }
            printVector(n->feed(input));
        }
    }
    return 0;
}

