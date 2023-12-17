#ifndef DIGITOR_NEURALNETWORK_H
#define DIGITOR_NEURALNETWORK_H

#include "vector"
#include "string"
#include "TrainData.h"
#include "nlohmann/json.hpp"

class NeuralNetwork {
public:
    explicit NeuralNetwork(const std::vector<unsigned int> &layers, const char *activationFn);

    explicit NeuralNetwork(const std::string &filename);

    std::string toString();

    std::vector<double> feed(const std::vector<double> &input);

    void train(const std::vector<TrainData> &data, unsigned int iterations, double learningRate);

private:
    void feedForward();

    void backPropagate(double cost, std::vector<double> target, double learningRate);

    void initJsonFile();

    void initRandom();

    void updateJsonFile();

    nlohmann::json readJsonFile();

    void setActivationType(int v);

    static long double ReLU(double v);

    static long double sigmoid(double v);

    [[nodiscard]] long double activationFn(double v) const;

    [[nodiscard]] long double activationFnDerivative(double v);

    int activationType{};
    std::string filename;
    std::vector<unsigned int> layers;
    std::vector<std::vector<double>> neuron;
    std::vector<std::vector<double>> bias;
    std::vector<std::vector<std::vector<double>>> weight;

    void writeJsonFile();

    double calculateCost(unsigned int targetValue);

    long double sigmoidDerivative(double v);

    long double ReLUDerivative(double v);
};


#endif //DIGITOR_NEURALNETWORK_H
