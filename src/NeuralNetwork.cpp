#include <random>
#include <utility>
#include "NeuralNetwork.h"
#include "string"
#include "fstream"
#include "iostream"
#include "nlohmann/json.hpp"

using json = nlohmann::json;

NeuralNetwork::NeuralNetwork(const std::vector<int> &neuronsPerLayer) {
    neuron.resize(neuronsPerLayer.size());
    weight.resize(neuronsPerLayer.size() - 1);
    bias.resize(neuronsPerLayer.size());
    for (int i = 0; i < neuronsPerLayer.size(); ++i) {
        neuron[i].resize(neuronsPerLayer[i]);
        bias[i].resize(neuronsPerLayer[i]);
    }
    for (int i = 0; i < weight.size(); ++i) {
        weight[i].resize(neuron[i + 1].size());
        for (int j = 0; j < weight[i].size(); ++j) {
            weight[i][j].resize(neuron[i].size());
        }
    }
    initRandom();
}

NeuralNetwork::NeuralNetwork(const std::string &filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open the JSON file." << std::endl;
        exit(EXIT_FAILURE);
    }
    std::string jsonString((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();
    json data = json::parse(jsonString);
    int numNeurons = data["layer"].size();
    neuron.resize(numNeurons);
    weight.resize(numNeurons - 1);
    bias.resize(numNeurons);
    for (int i = 0; i < numNeurons; ++i) {
        neuron[i].resize(data["layer"][i]);
        bias[i].resize(data["layer"][i]);
    }
    for (int i = 0; i < weight.size(); ++i) {
        weight[i].resize(neuron[i + 1].size());
        for (int j = 0; j < weight[i].size(); ++j) {
            weight[i][j].resize(neuron[i].size());
        }
    }
    for (int i = 0; i < weight.size(); ++i) {
        for (int j = 0; j < weight[i].size(); ++j) {
            for (int k = 0; k < weight[i][j].size(); ++k) {
                weight[i][j][k] = data["weights"][i][j][k];
            }
        }
    }
    for (int i = 0; i < bias.size(); ++i) {
        for (int j = 0; j < bias[i].size(); ++j) {
            bias[i][j] = data["biases"][i][j];
        }
    }

}

void NeuralNetwork::feed(const std::vector<double> &input) {
    if (input.size() != neuron[0].size()) exit(EXIT_FAILURE);
    for (int i = 0; i < input.size(); ++i) {
        neuron[0][i] = input[i];
    }
    feedForward();
    for (double i: neuron[neuron.size() - 1]) {
        std::cout << i << std::endl;
    }
}

void NeuralNetwork::feedForward() {
    for (int i = 1; i < neuron.size(); ++i) {
        for (int j = 0; j < neuron[i].size(); ++j) {
            for (int k = 0; k < neuron[i - 1].size(); ++k) {
                neuron[i][j] += weight[i - 1][j][k] * neuron[i - 1][k];
            }
            if (i != neuron.size() - 1) neuron[i][j] = activationFn(neuron[i][j] + bias[i][j]);
        }
    }
}

double NeuralNetwork::ReLU(double v) {
    return v > 0 ? v : 0;
}

double NeuralNetwork::sigmoid(double v) {
    return 1.0 / (1.0 + exp(-v));
}

double NeuralNetwork::activationFn(double v) {
    return sigmoid(v);
}

void NeuralNetwork::initRandom() {
    for (auto &i: weight) {
        for (auto &j: i) {
            for (double &k: j) {
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_real_distribution<> dis(-2.0, 2.0);
                k = dis(gen);
            }
        }
    }
    for (auto &bia: bias) {
        for (double &j: bia) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> dis(-1.0, 1.0);
            j = dis(gen);
        }
    }
}

std::string NeuralNetwork::toString() {
    std::string result;
    result += "neurons: \n";
    for (auto &i: neuron) {
        for (double j: i) {
            result += std::to_string(j);
            result += " ";
        }
        result += "\n";
    }
    result += "weights: \n";
    for (auto &i: weight) {
        for (auto &j: i) {
            for (double k: j) {
                result += std::to_string(k);
                result += " ";
            }
            result += "\n";
        }
        result += "\n\n";
    }
    return result;
}