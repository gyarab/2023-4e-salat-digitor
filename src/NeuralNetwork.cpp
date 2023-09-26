//
// Created by ondrej on 26/09/23.
//

#include "NeuralNetwork.h"
#include "string"

NeuralNetwork::NeuralNetwork(std::vector<Layer> layers) {
    this->layers = layers;
}

std::string NeuralNetwork::toString() {
    std::string result;
    for (int i = 0; i < layers.size(); i++) {
        result += "Layer " + std::to_string(i) + " : " + std::to_string(layers[i].numNeurons) + "\n";
    }
    return result;
}
