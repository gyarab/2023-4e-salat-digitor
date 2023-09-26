//
// Created by ondrej on 26/09/23.
//

#include "Connection.h"

Connection::Connection(Neuron &fromNeuron, Neuron &toNeuron) {
    this->fromNeuron = fromNeuron;
    this->toNeuron = toNeuron;
}

double Connection::getOutput() {
    return weight * getInput();
}

double Connection::getInput() {
    return fromNeuron.getOutput();
}

double Connection::getWeight() {
    return weight;
}

void Connection::setWeight(double weight) {
    this->weight = weight;
}
