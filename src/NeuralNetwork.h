#ifndef DIGITOR_NEURALNETWORK_H
#define DIGITOR_NEURALNETWORK_H

#include "vector"
#include "string"

class NeuralNetwork {
public:
    explicit NeuralNetwork(const std::vector<int> &neuronsPerLayer);

    explicit NeuralNetwork(const std::string &filename);

    std::string toString();

    void feed(const std::vector<double> &input);


private:
    void feedForward();

    void initRandom();

    static double ReLU(double v);

    std::vector<std::vector<double>> neuron;
    std::vector<std::vector<double>> bias;
    std::vector<std::vector<std::vector<double>>> weight;
};


#endif //DIGITOR_NEURALNETWORK_H
