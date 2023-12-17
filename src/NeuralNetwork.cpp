#include <random>
#include "NeuralNetwork.h"
#include "string"
#include "fstream"
#include "iostream"
#include "TrainData.h"
#include "nlohmann/json.hpp"

using json = nlohmann::json;

NeuralNetwork::NeuralNetwork(const std::vector<unsigned int> &neuronsPerLayer, const char *activationFn = "sigmoid") {
    if (strcmp(activationFn, "sigmoid") == 0) setActivationType(0);
    else if (strcmp(activationFn, "ReLU") == 0) setActivationType(1);
    else setActivationType(-1);
    layers = neuronsPerLayer;
    neuron.resize(layers.size());
    weight.resize(layers.size() - 1);
    bias.resize(layers.size());
    for (int i = 0; i < layers.size(); ++i) {
        neuron[i].resize(layers[i]);
        bias[i].resize(layers[i]);
    }
    for (int i = 0; i < weight.size(); ++i) {
        weight[i].resize(neuron[i + 1].size());
        for (int j = 0; j < weight[i].size(); ++j) {
            weight[i][j].resize(neuron[i].size());
        }
    }
    initRandom();
    initJsonFile();
}

NeuralNetwork::NeuralNetwork(const std::string &file) {
    this->filename = file;
    json data = readJsonFile();
    setActivationType(data["activation"]);
    auto jLayers = data["layer"];
    auto numNeurons = jLayers.size();
    for (const auto &nLayer: jLayers) {
        layers.push_back(nLayer);
    }
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

std::vector<double> NeuralNetwork::feed(const std::vector<double> &input) {
    if (input.size() != neuron[0].size()) {
        std::cerr << "Wrong input format" << std::endl;
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < input.size(); ++i) {
        neuron[0][i] = input[i];
    }
    feedForward();
    std::vector<double> result;
    for (double i: neuron[neuron.size() - 1]) {
        result.push_back(i);
    }
    return result;
}

void NeuralNetwork::feedForward() {
    for (int i = 1; i < neuron.size(); ++i) {
        for (int j = 0; j < neuron[i].size(); ++j) {
            neuron[i][j] = 0;
        }
    }
    for (int i = 1; i < neuron.size(); ++i) {
        for (int j = 0; j < neuron[i].size(); ++j) {
            for (int k = 0; k < neuron[i - 1].size(); ++k) {
                neuron[i][j] += weight[i - 1][j][k] * neuron[i - 1][k];
            }
            neuron[i][j] = activationFn(neuron[i][j] + bias[i][j]);
        }
    }
}

void NeuralNetwork::train(const std::vector<TrainData> &data, unsigned int iterations, double learningRate) {
    double totalCost;
    for (int i = 0; i < iterations; ++i) {
        for (const auto &d: data) {
            if (d.image.size() != neuron[0].size()) {
                std::cerr << "Wrong input format" << std::endl;
                exit(EXIT_FAILURE);
            }
            for (int j = 0; j < d.image.size(); ++j) {
                neuron[0][j] = d.image[j];
            }
            feedForward();
            totalCost = calculateCost(d.value);
            std::vector<double> target;
            target.resize(layers[layers.size() - 1]);
            for (int j = 0; j < target.size(); ++j) {
                if (j == d.value) target[j] = 1;
                else target[j] = 0;
            }
            backPropagate(totalCost, target, learningRate);

        }
    }
    updateJsonFile();
}

void NeuralNetwork::backPropagate(double cost, std::vector<double> target, double learningRate) {
    std::vector<double> relativeDeltaErrors(layers[layers.size() - 1]);
    int lastLayerIndex = weight.size() - 1;
    int lastLayerNeuronsIndex = (neuron.size() - 1) - ((weight.size() - 1) - lastLayerIndex);
    /**
     * back propagate last layer
     */
    for (int j = 0; j < weight[lastLayerIndex].size(); ++j) {
        for (int k = 0; k < weight[lastLayerIndex][j].size(); ++k) {
            double outputO1 = neuron[lastLayerNeuronsIndex][j];
            double outputH1 = neuron[lastLayerNeuronsIndex - 1][k];
            double localCost = (outputO1 - target[j]) * activationFnDerivative(outputO1) * outputH1;
            weight[lastLayerIndex][j][k] -= learningRate * localCost;
        }
    }
    /**
     * back propagate the rest
     */
    for (int i = lastLayerIndex - 1; i >= 0; --i) {
        for (int j = 0; j < weight[i].size(); ++j) {
            for (int k = 0; k < weight[i][j].size(); ++k) {
                long double localCost = 0;
                for (int l = 0; l < relativeDeltaErrors.size(); ++l) {
                    relativeDeltaErrors[l] = (neuron[lastLayerNeuronsIndex][l] - target[l]) * weight[i + 1][l][j] *
                                             activationFnDerivative(neuron[i + 1][k]) * neuron[i][k] *
                                             activationFnDerivative(neuron[lastLayerNeuronsIndex][l]);
                    if (std::isnan(relativeDeltaErrors[l])) {
                        std::cout << activationFnDerivative(neuron[i + 1][k]) << std::endl;
                        std::cout << neuron[i + 1][k] << std::endl;
                        std::cout << neuron[i + 1][k] * (1.0 - neuron[i + 1][k]) << std::endl;
                        std::cout << 1 - neuron[i + 1][k] << std::endl;
                        exit(69);
                    }
                    localCost += relativeDeltaErrors[l];
                }
                weight[i][j][k] -= learningRate * localCost;
                if (std::isnan(weight[i][j][k])) {
                    std::cout << i << " " << j << " " << k << std::endl;
                    std::cout << learningRate << " " << localCost << std::endl;
                    exit(-1);
                }
            }
        }
    }
}

double NeuralNetwork::calculateCost(unsigned int targetValue) {
    double cost = 0;
    for (int i = 0; i < neuron[neuron.size() - 1].size(); ++i) {
        if (i == targetValue) cost += pow(2, (1 - neuron[neuron.size() - 1][i]));
        else cost += pow(2, (0 - neuron[neuron.size() - 1][i]));
    }
    return cost;
}

long double NeuralNetwork::ReLU(double v) {
    return v > 0 ? v : 0;
}

long double NeuralNetwork::ReLUDerivative(double v) {
    return v > 0 ? 1 : 0;
}

long double NeuralNetwork::sigmoid(double v) {
    return 1.0 / (1.0 + exp(-v));
}

long double NeuralNetwork::sigmoidDerivative(double v) {
    long double result = v * (1.0 - v);
    if (std::isinf(result)) {
        return v > 0 ? 1 : -1;
    }
    return result;
}


long double NeuralNetwork::activationFn(double v) const {
    switch (activationType) {
        case 0:
            return sigmoid(v);
        case 1:
            return ReLU(v);
        default:
            std::cerr << "Invalid activation function" << std::endl;
            exit(EXIT_FAILURE);
    }
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
    for (double &i: bias[bias.size() - 1]) {
        i = 0;
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

void NeuralNetwork::initJsonFile() {
    std::string file;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1.0);
    file +=
            "neuralNetwork(layers=" + std::to_string(neuron.size()) + ", id=" +
            std::to_string((int) round(dis(gen) * 10000)) +
            ").json";
    std::ofstream output_file(file);
    filename = file;
    writeJsonFile();
}

void NeuralNetwork::updateJsonFile() {
    writeJsonFile();
}

void NeuralNetwork::writeJsonFile() {
    std::ofstream output_file(filename);
    if (output_file.is_open()) {
        json data;
        data["activation"] = activationType;
        data["layer"] = layers;
        data["weights"] = weight;
        data["biases"] = bias;
        std::string stringData = data.dump(4);
        output_file << data.dump(4);
        output_file.close();
        std::cout << "JSON data saved to '" << filename << "'" << std::endl;
    } else {
        std::cerr << "Failed to open the output file." << std::endl;
    }
}

nlohmann::json NeuralNetwork::readJsonFile() {
    std::ifstream jFile(filename);
    if (!jFile.is_open()) {
        std::cerr << "Failed to open the JSON file." << std::endl;
        exit(EXIT_FAILURE);
    }
    std::string jsonString((std::istreambuf_iterator<char>(jFile)), std::istreambuf_iterator<char>());
    jFile.close();
    return json::parse(jsonString);
}

void NeuralNetwork::setActivationType(int v) {
    this->activationType = v;
}

long double NeuralNetwork::activationFnDerivative(double v) {
    switch (activationType) {
        case 0:
            return sigmoidDerivative(v);
        case 1:
            return ReLUDerivative(v);
        default:
            std::cerr << "Invalid activation function" << std::endl;
            exit(EXIT_FAILURE);
    }
}



