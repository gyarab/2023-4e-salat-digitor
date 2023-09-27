//
// Created by ondrej on 26/09/23.
//

#include "Neuron.h"
#include "cmath"
#include "random"

Neuron::Neuron() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 10.0);
    this->bias = dis(gen);
}

double Neuron::getBias() const {
    return this->bias;
}

void Neuron::setBias(double b) {
    this->bias = b;
}

double Neuron::getValue() const {
    return value;
}

double Neuron::getOutput() const {
    return sigmoid(getValue() + bias);
}

double Neuron::sigmoid(double v) {
    return 1 / (1 + exp(-v));
}

void Neuron::setValue(double v) {
    this->value = v;
}

void Neuron::addValue(double v) {
    this->value += v;
}

std::string Neuron::toString() const {
    return "Neuron : bias " + std::to_string(bias) + std::to_string(sigmoid(value));
};



