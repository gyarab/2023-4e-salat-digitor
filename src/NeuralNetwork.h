#ifndef DIGITOR_NEURALNETWORK_H
#define DIGITOR_NEURALNETWORK_H

#include "vector"
#include "string"
#include "nlohmann/json.hpp"

class NeuralNetwork {
public:
    explicit NeuralNetwork(const std::vector<int> &layers);

    explicit NeuralNetwork(const std::string &filename);

    std::string toString();

    void feed(const std::vector<double> &input);


private:
    void feedForward();

    void initJsonFile();

    void initRandom();

    void updateJsonFile();

    nlohmann::json readJsonFile();

    static double ReLU(double v);

    static double sigmoid(double v);

    static double activationFn(double v);

    std::string filename;
    std::vector<int> layers;
    std::vector<std::vector<double>> neuron;
    std::vector<std::vector<double>> bias;
    std::vector<std::vector<std::vector<double>>> weight;

    void writeJsonFile();
};


#endif //DIGITOR_NEURALNETWORK_H
