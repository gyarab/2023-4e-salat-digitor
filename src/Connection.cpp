//
// Created by ondrej on 26/09/23.
//

#include "Connection.h"
#include "random"
#include "iostream"

Connection::Connection(Neuron *fromNeuron, Neuron *toNeuron) : fromNeuron(fromNeuron), toNeuron(toNeuron) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    this->weight = dis(gen);
}

double Connection::getOutput() const {
    return weight * getInput();
}

double Connection::getInput() const {
    return fromNeuron->getOutput();
}

double Connection::getWeight() const {
    return weight;
}

void Connection::setWeight(double weight) {
    this->weight = weight;
}
