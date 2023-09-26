//
// Created by ondrej on 26/09/23.
//

#include "Neuron.h"
#include "cmath"

Neuron::Neuron() {}

double Neuron::getBias() const {
    return bias;
}

void Neuron::setBias(double bias) {
    this->bias = bias;
}

double Neuron::getOutput() const {
    return 1 / (1 + exp(-value));
}

void Neuron::addValue(double value) {
    this->value += value;
}

std::string Neuron::toString() const {
    return "Neuron : bias " + std::to_string(bias) + std::to_string(getOutput());
};

