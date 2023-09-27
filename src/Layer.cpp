//
// Created by ondrej on 26/09/23.
//

#include "Layer.h"
#include "iostream"
#include "Neuron.h"
#include "Connection.h"

Layer::Layer(int numNeurons) {
    for (int i = 0; i < numNeurons; i++) {
        Neuron *n = new Neuron();
        this->neurons.push_back(n);
        neurons[i]->setBias(i);
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

void Layer::feedForward(bool input) {
    if (input) {
        for (auto &connection: connections) {
            connection.toNeuron->value += (connection.getWeight() * connection.fromNeuron->value);
        }
    } else {
        for (auto &connection: connections) {
            connection.toNeuron->value += (connection.getWeight() * connection.fromNeuron->value);
        }
    }
}

std::string Layer::toString() {
    std::string result;
    for (int i = 0; i < connections.size(); i++) {
        result +=
                "\tConnection: " + std::to_string(i) + " neuron " + std::to_string(connections[i].fromNeuron->getBias())
                + " -> " + std::to_string(connections[i].toNeuron->getBias()) + "\n";
    }
    return result;
}