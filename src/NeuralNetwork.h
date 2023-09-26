//
// Created by ondrej on 26/09/23.
//

#ifndef DIGITOR_NEURALNETWORK_H
#define DIGITOR_NEURALNETWORK_H

#include "vector"
#include "string"
#include "Layer.h"

class NeuralNetwork {
public:
    NeuralNetwork(std::vector<Layer> layers);

    std::string toString();

    std::vector<Layer> layers;

};


#endif //DIGITOR_NEURALNETWORK_H
