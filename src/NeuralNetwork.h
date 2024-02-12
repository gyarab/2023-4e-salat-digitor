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

    std::vector<unsigned int> layers;

    std::string toString();

    std::vector<double> feed(const std::vector<double> &input);

    void train(const std::vector<std::vector<TrainData>> &data, unsigned int iterations, long double learningRate);

    void saveProgress();

private:
    void feedForward();

    void backPropagate(std::vector<double> target, long double learningRate,
                       std::vector<std::vector<std::vector<long double>>> &newWeights);

    void initJsonFile();

    void initRandom();

    void updateJsonFile();

    nlohmann::json readJsonFile();

    void setActivationType(int v);

    static long double ReLU(long double v);

    static long double sigmoid(long double v);

    [[nodiscard]] long double activationFn(long double v) const;

    [[nodiscard]] long double activationFnDerivative(long double v) const;

    int activationType{};
    std::string filename;
    std::vector<std::vector<long double>> neuron;
    std::vector<std::vector<long double>> rawNeuron;
    std::vector<std::vector<long double>> bias;
    std::vector<std::vector<std::vector<long double>>> weight;


    void writeJsonFile(std::vector<std::vector<std::vector<long double>>> &weightToSave,
                       std::vector<std::vector<long double>> &biasToSave, bool print);

    double calculateCost(unsigned int targetValue);

    static long double sigmoidDerivative(long double v);

    static long double ReLUDerivative(long double v);
};


#endif //DIGITOR_NEURALNETWORK_H
