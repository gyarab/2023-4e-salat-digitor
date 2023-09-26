//
// Created by ondrej on 26/09/23.
//

#ifndef DIGITOR_CONNECTION_H
#define DIGITOR_CONNECTION_H

#include "Neuron.h"

class Connection {
public:
    Connection(Neuron &fromNeuron, Neuron &toNeuron);

    void setWeight(double weight);

    double getWeight();

    double getInput();

    double getOutput();

    Neuron fromNeuron;
    Neuron toNeuron;
    double weight{};
};


#endif //DIGITOR_CONNECTION_H
