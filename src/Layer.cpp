//
// Created by ondrej on 26/09/23.
//

#include "Layer.h"
#include "iostream"
#include "Neuron.h"
#include "Connection.h"

Layer::Layer(int numNeurons) {
    for (int i = 0; i < numNeurons; i++) {
        neurons.push_back(Neuron());
        neurons[i].setBias(i);
    }
    this->numNeurons = numNeurons;
}

void Layer::connectLayer(Layer &nextLayer) {
    for (int i = 0; i < this->numNeurons; ++i) {
        for (int j = 0; j < nextLayer.numNeurons; ++j) {
            this->connections.push_back(Connection(this->neurons[i], nextLayer.neurons[j]));
        }
    }
}

void Layer::feedForward() {

}

std::string Layer::toString() {
    std::string result;
    for (int i = 0; i < connections.size(); i++) {
        int *ptr1 = reinterpret_cast<int *>(&connections[i].fromNeuron);
        int *ptr2 = reinterpret_cast<int *>(&connections[i].toNeuron);
        result +=
                "\tConnection: " + std::to_string(i) + " neuron " + std::to_string(connections[i].fromNeuron.getBias())
                + " -> " + std::to_string(connections[i].toNeuron.getBias()) + "\n";
    }
    return result;
}
