//
// Created by ondrej on 26/09/23.
//

#ifndef DIGITOR_CONNECTION_H
#define DIGITOR_CONNECTION_H

#include "Neuron.h"

class Connection {
public:
    Connection(Neuron *fromNeuron, Neuron *toNeuron);

    Connection &operator=(const Connection &) {
        return *this;
    }

    void setWeight(double weight);

    double getWeight() const;

    double getInput() const;

    double getOutput() const;

    Neuron *fromNeuron;
    Neuron *toNeuron;
    double weight{};
};


#endif //DIGITOR_CONNECTION_H
