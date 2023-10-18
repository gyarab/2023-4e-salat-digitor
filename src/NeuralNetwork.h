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

    void train(const std::vector<TrainData> &data);

private:
    void feedForward();

    std::vector<std::vector<std::vector<double>>> backPropagate();

    void initJsonFile();

    void initRandom();

    void updateJsonFile();

    nlohmann::json readJsonFile();

    void setActivationType(int v);

    static double ReLU(double v);

    static double sigmoid(double v);

    [[nodiscard]] double activationFn(double v) const;

    int activationType{};
    std::string filename;
    std::vector<unsigned int> layers;
    std::vector<std::vector<double>> neuron;
    std::vector<std::vector<double>> bias;
    std::vector<std::vector<std::vector<double>>> weight;

    void writeJsonFile();

    double calculateCost(unsigned int targetValue);
};


#endif //DIGITOR_NEURALNETWORK_H
