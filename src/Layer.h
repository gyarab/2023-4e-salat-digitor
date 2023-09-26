//
// Created by ondrej on 26/09/23.
//

#ifndef DIGITOR_LAYER_H
#define DIGITOR_LAYER_H

#include "iostream"
#include "vector"
#include "Neuron.h"
#include "Connection.h"

class Layer {
public:
    Layer(int numNeurons);

    void connectLayer(Layer &nextLayer);

    void feedForward();

    int numNeurons{};
    std::vector<Neuron> neurons;
    std::vector<Connection> connections;

};


#endif //DIGITOR_LAYER_H
