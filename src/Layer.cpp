//
// Created by ondrej on 26/09/23.
//

#include "Layer.h"
#include "Neuron.h"
#include "Connection.h"

Layer::Layer(int numNeurons) {
    for (int i = 0; i < numNeurons; i++) {
        neurons.push_back(Neuron());
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
