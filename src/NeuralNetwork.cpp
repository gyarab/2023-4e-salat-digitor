#include <random>
#include "NeuralNetwork.h"
#include "string"
#include "fstream"
#include "iostream"
#include "TrainData.h"
#include "nlohmann/json.hpp"

#include <omp.h>
#include "mutex"

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
    rawNeuron = neuron;
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
    rawNeuron = neuron;
}

std::vector<double> NeuralNetwork::feed(const std::vector<double> &input) {
    if (input.size() != neuron[0].size()) {
        std::cerr << "Wrong input format" << std::endl;
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < input.size(); ++i) {
        neuron[0][i] = input[i];
        rawNeuron[0][i] = input[i];
    }
    feedForward();
    std::vector<double> result;
    for (long double i: neuron[neuron.size() - 1]) {
        result.push_back((double) i);
    }
    return result;
}

void NeuralNetwork::feedForward() {
    for (int i = 1; i < neuron.size(); ++i) {
        for (int j = 0; j < neuron[i].size(); ++j) {
            neuron[i][j] = 0;
            rawNeuron[i][j] = 0;
        }
    }
    for (int i = 1; i < neuron.size(); ++i) {
        for (int j = 0; j < neuron[i].size(); ++j) {
            for (int k = 0; k < neuron[i - 1].size(); ++k) {
                rawNeuron[i][j] += weight[i - 1][j][k] * neuron[i - 1][k];
            }
            neuron[i][j] = activationFn(rawNeuron[i][j] + bias[i][j]);
        }
    }
}

void NeuralNetwork::train(const std::vector<std::vector<TrainData>> &data, unsigned int iterations,
                          long double learningRate) {
    std::vector<std::vector<std::vector<long double>>> minCostWeights;
    std::vector<std::vector<std::vector<long double>>> newWeights;
    std::vector<std::vector<long double>> newBiases;
    std::vector<std::vector<long double>> relativeDeltaErrors;
    relativeDeltaErrors.resize(layers.size());
    for (int j = 0; j < layers.size(); ++j) {
        relativeDeltaErrors[j].resize(layers[j], 0);
    }
    double minCost = -1;
    for (int i = 0; i < iterations; ++i) {
        double cost = 0;
        newWeights = weight;
        newBiases = neuron;
        for (auto &set: data) {
            for (const auto &d: set) {
                if (d.image.size() != neuron[0].size()) {
                    std::cerr << "Wrong input format" << std::endl;
                    exit(EXIT_FAILURE);
                }
                for (int j = 0; j < d.image.size(); ++j) {
                    neuron[0][j] = d.image[j];
                }
                feedForward();
                cost += calculateCost(d.value);
                std::vector<double> target;
                target.resize(layers[layers.size() - 1]);
                for (int j = 0; j < target.size(); ++j) {
                    if (j == d.value) target[j] = 1;
                    else target[j] = 0;
                }
                backPropagate(target, relativeDeltaErrors, learningRate, newWeights, newBiases);
            }
            weight = newWeights;
        }
        double progress = (double) i * 100 / iterations;
        double totalCost = cost / ((double) data.size() * (double) data[0].size());
        if (totalCost < minCost || minCost == -1) {
            minCost = totalCost;
            minCostWeights = weight;
        }
        std::cout << "\rProgress: " << std::fixed << std::setprecision(2) << progress << "% | " << "Total cost: "
                  << std::fixed << std::setprecision(8) << totalCost << std::flush;
        if (i % 1000 == 0) saveProgress();
    }
    updateJsonFile();
}

void NeuralNetwork::backPropagate(std::vector<double> target,
                                  std::vector<std::vector<long double>> relativeDeltaErrors,
                                  long double learningRate,
                                  std::vector<std::vector<std::vector<long double>>> &newWeights,
                                  std::vector<std::vector<long double>> &newBiases) {
    unsigned int lastLayerIndex = weight.size() - 1;
    unsigned int lastLayerNeuronsIndex = (neuron.size() - 1) - ((weight.size() - 1) - lastLayerIndex);
    /**
     * back propagate last layer
     *
     * using chain rule formula:
     * localCost = ∂Error/∂w = ∂Error/∂neuron * ∂neuron/∂rewNeuron * ∂neuron/∂weight
     * = 2*(output * target) * (rawNeuron)' * neuron
     *
     * newWeight -= localCost * learningRate
     *
     */
    for (int i = 0; i < weight[lastLayerIndex].size(); ++i) {
        long double output = neuron[lastLayerNeuronsIndex][i];
        relativeDeltaErrors[lastLayerNeuronsIndex][i] = 2 * (output - target[i]);
        for (int j = 0; j < weight[lastLayerIndex][i].size(); ++j) {
            long double source = neuron[lastLayerNeuronsIndex - 1][j];
            long double rawSource = rawNeuron[lastLayerNeuronsIndex - 1][j];
            long double localError = relativeDeltaErrors[lastLayerNeuronsIndex][i] * activationFnDerivative(rawSource);
            newWeights[lastLayerIndex][i][j] -= learningRate * localError * source;
            newBiases[lastLayerIndex][j] -= learningRate * localError;
        }
    }

    /**
     * back propagate the rest
     *
     * because the weight in hidden layers are (from definition) connected to all neuron in the next layers
     * we need to first calculate the ∂Error/∂neuron = sum( ∂Error/∂next_neurons *  weight * raw_neuron )
     *
     * then the we can calculate the localCost using the very same formula as we used in backpropagation the last layer
     *
     */
    for (int i = (int) (lastLayerIndex - 1); i >= 0; --i) {
        for (int j = 0; j < weight[i].size(); ++j) {
            long double subtotal = 0;
            for (int l = 0; l < relativeDeltaErrors[i + 2].size(); ++l) {
                subtotal += relativeDeltaErrors[i + 2][l] * activationFnDerivative(rawNeuron[i + 1][j]) *
                            weight[i + 1][l][j];
            }
            relativeDeltaErrors[i + 1][j] = subtotal;
            long double localError = activationFnDerivative(rawNeuron[i + 1][j]) * relativeDeltaErrors[i + 1][j];
            newBiases[i + 1][j] -= learningRate * localError;
            for (int k = 0; k < weight[i][j].size(); ++k) {
                newWeights[i][j][k] -= learningRate * localError * neuron[i][k];
            }
        }
    }
}

double NeuralNetwork::calculateCost(unsigned int targetValue) {
    double cost = 0;
    for (int i = 0; i < neuron[neuron.size() - 1].size(); ++i) {
        if (i == targetValue) {
            cost += pow((1 - (double) neuron[neuron.size() - 1][i]), 2);
        } else {
            cost += pow((0 - (double) neuron[neuron.size() - 1][i]), 2);
        }
    }
    return cost;
}

long double NeuralNetwork::ReLU(long double v) {
    return v > 0 ? v : 0;
}

long double NeuralNetwork::ReLUDerivative(long double v) {
    return v > 0 ? 1 : 0;
}

long double NeuralNetwork::sigmoid(long double v) {
    if (v >= 650) {
        return 1.0;
    } else if (v <= -650) {
        return 0.0;
    }
    return (1.0 / (1.0 + exp(-(double) v)));
}

long double NeuralNetwork::sigmoidDerivative(long double v) {
    long double sig = sigmoid(v);
    long double result = sig * (1.0 - sig);
    return result;
}


long double NeuralNetwork::activationFn(long double v) const {
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

void NeuralNetwork::setActivationType(int v) {
    this->activationType = v;
}

long double NeuralNetwork::activationFnDerivative(long double v) const {
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

void NeuralNetwork::initRandom() {
    for (auto &i: weight) {
        for (auto &j: i) {
            for (long double &k: j) {
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_real_distribution<> dis(-1.0, 1.0);
                k = dis(gen);
            }
        }
    }
    for (auto &bia: bias) {
        for (long double &j: bia) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> dis(-0.1, 0.1);
            j = dis(gen);
        }
    }
    for (long double &i: bias[bias.size() - 1]) {
        i = 0;
    }
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
    writeJsonFile(weight, bias, true);
}


void NeuralNetwork::saveProgress() {
    writeJsonFile(weight, bias, false);
}


void NeuralNetwork::updateJsonFile() {
    writeJsonFile(weight, bias, true);
}


void NeuralNetwork::writeJsonFile(std::vector<std::vector<std::vector<long double>>> &weightToSave,
                                  std::vector<std::vector<long double>> &biasToSave, bool print) {
    std::ofstream output_file(filename);
    if (output_file.is_open()) {
        json data;
        data["activation"] = activationType;
        data["layer"] = layers;
        data["weights"] = weightToSave;
        data["biases"] = biasToSave;
        std::string stringData = data.dump(4);
        output_file << data.dump(4);
        output_file.close();
        if (print) std::cout << "JSON data saved to '" << filename << "'" << std::endl;
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


std::string NeuralNetwork::toString() {
    std::string result;
    result += "neurons: \n";
    for (auto &i: neuron) {
        for (long double j: i) {
            result += std::to_string(j);
            result += " ";
        }
        result += "\n";
    }
    result += "weights: \n";
    for (auto &i: weight) {
        for (auto &j: i) {
            for (long double k: j) {
                result += std::to_string(k);
                result += " ";
            }
            result += "\n";
        }
        result += "\n\n";
    }
    return result;
}
