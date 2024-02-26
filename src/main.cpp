#include "NeuralNetwork.h"
#include "iostream"
#include "TrainData.h"
#include<bits/stdc++.h>

using namespace std;

NeuralNetwork *n = nullptr;
bool train = false; // -t

void signalHandler(int signal) {
    if (signal == SIGINT) {
        std::cout << "\nCtrl+C detected. Saving progress and exiting...\n";
        if (train) {
            n->saveProgress();
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


int main(int argc, char *argv[]) {

    std::signal(SIGINT, signalHandler);

    if (argc < 2) {
        cerr << "Invalid output: Must contain at least one argument with the name of neural network file." << endl;
    }

    bool newFile = false; // -n
    const char *filename;
    const char *activationFn;
    vector<unsigned int> neuronsPerLayer;
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
        neuronsPerLayer = charArrToVector(argv[3]);
        iterations = stoi(argv[4]);
        learningRate = stod(argv[5]);
        activationFn = argv[6];
        batches = stoi(argv[7]);
        data_size = stoi(argv[8]);
        n = new NeuralNetwork(neuronsPerLayer, activationFn);
        vector<vector<TrainData>> trainData(batches);
        for (int i = 0; i < batches; ++i) {
            for (int j = 0; j < data_size; ++j) {
                vector<double> input(neuronsPerLayer[0]);
                for (int k = 0; k < neuronsPerLayer[0]; ++k) {
                    cin >> input[k];
                }
                unsigned int target;
                cin >> target;
                trainData[i].push_back({input, target});
            }
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
        n = new NeuralNetwork(filename);
        vector<vector<TrainData>> trainData(batches);
        for (int i = 0; i < batches; ++i) {
            for (int j = 0; j < data_size; ++j) {
                vector<double> input(n->layers[0]);
                for (int k = 0; k < n->layers[0]; ++k) {
                    cin >> input[k];
                }
                int target;
                cin >> target;
                trainData[i].push_back({input, static_cast<unsigned int>(target)});
            }
        }
        n->train(trainData, iterations, learningRate);
    } else if (newFile) {
        if (argc < 4) {
            cerr << "Missing arguments" << endl
                 << "Expected: `./digitor -n <neurons_per_layer> <activation_function>`"
                 << endl;
            exit(EXIT_FAILURE);
        }
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
        if (argc < 2) {
            cerr << "Missing arguments" << endl
                 << "Expected: `./digitor <filename>`"
                 << endl;
            exit(EXIT_FAILURE);
        }
        filename = argv[1];
        n = new NeuralNetwork(filename);
        while (true) {
            vector<double> input(n->layers[0]);
            for (int i = 0; i < n->layers[0]; ++i) {
                cin >> input[i];
            }
            vector<double> output = n->feed(input);
            //printVector(output);
            unsigned int estimation = max_element(output.begin(), output.end()) - output.begin();
            /*cout << estimation << " -> " << std::fixed << std::setprecision(2) << output[estimation] * 100 << "%"
                 << endl;*/
            cout << estimation << endl;
        }
    }
    return 0;
}

