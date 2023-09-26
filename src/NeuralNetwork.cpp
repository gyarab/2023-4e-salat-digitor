//
// Created by ondrej on 26/09/23.
//

#include "NeuralNetwork.h"
#include "string"
#include "iostream"

NeuralNetwork::NeuralNetwork(std::vector<Layer> layers) {
    this->layers = layers;
    for (int i = 0; i < this->layers.size() - 1; i++) {
        this->layers[i].connectLayer(layers[i + 1]);
    }
}

std::string NeuralNetwork::toString() {
    std::string result;
    for (int i = 0; i < layers.size(); i++) {
        result += "Layer " + std::to_string(i) + " : " + std::to_string(layers[i].neurons.size()) + "\n" +
                  " " + layers[i].toString();
    }
    return result;
}
