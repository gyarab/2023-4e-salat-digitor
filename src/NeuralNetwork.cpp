#include <random>
#include <utility>
#include "NeuralNetwork.h"
#include "string"
#include "fstream"
#include "iostream"
#include "TrainData.h"
#include "nlohmann/json.hpp"
#include "atomic"
#include "vector"
#include "thread"
#include "queue"
#include "condition_variable"
#include "functional"

using json = nlohmann::json;

struct TrainingTask {
    NeuralNetwork n;
    TrainData dataSubset;
    std::vector<std::vector<std::vector<std::atomic<double>>>> &T_newWeights;
    std::vector<std::vector<double>> T_relavieDeltaErrors;
    std::atomic<double> &T_cost;
    double T_learningRate;
};


class ThreadPool {
public:
    explicit ThreadPool(int numThreads) : isRunning(true) {
        for (int i = 0; i < numThreads; ++i) {
            workers.emplace_back(&ThreadPool::workerThread, this);
        }
    }

    ~ThreadPool() {
        stop();
        for (auto &thread: workers) {
            thread.join();
        }
    }

    void addTask(const TrainingTask &task) {
        std::lock_guard<std::mutex> lock(mutex);
        tasks.push(task);
        condition.notify_one();
    }

    void waitAllTasks() {
        std::unique_lock<std::mutex> lock(mutex);
        while (!tasks.empty()) {
            condition.wait(lock);
        }
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            isRunning = false;
        }
        condition.notify_all();
    }

private:
    void workerThread() {
        while (isRunning) {
            std::unique_lock<std::mutex> lock(mutex);
            condition.wait(lock, [&] { return !isRunning || !tasks.empty(); });
            TrainingTask task = std::move(tasks.front());
            tasks.pop();
            task.n.trainSubset(task.dataSubset, task.T_newWeights, task.T_relavieDeltaErrors, task.T_cost,
                               task.T_learningRate);
        }
    }

    std::vector<std::thread> workers;
    std::queue<TrainingTask> tasks;
    std::mutex mutex;
    std::condition_variable condition;
    bool isRunning;
};

NeuralNetwork::NeuralNetwork(const std::vector<unsigned int> &neuronsPerLayer, const char *activationFn = "sigmoid") {
    if (strcmp(activationFn, "sigmoid") == 0) setActivationType(0);
    else if (strcmp(activationFn, "ReLU") == 0) setActivationType(1);
    else setActivationType(-1);
    layers = neuronsPerLayer;
    neuron.resize(layers.size());
    weight.resize(layers.size() - 1);
    bias.resize(layers.size());
    for (int i = 0; i < layers.size(); ++i) {
        neuron[i].resize(layers[i]);
        bias[i].resize(layers[i]);
    }
    for (int i = 0; i < weight.size(); ++i) {
        weight[i].resize(neuron[i + 1].size());
        for (int j = 0; j < weight[i].size(); ++j) {
            weight[i][j].resize(neuron[i].size());
        }
    }
    rawNeuron = neuron;
    initRandom();
    initJsonFile();
}

NeuralNetwork::NeuralNetwork(const std::string &file) {
    this->filename = file;
    json data = readJsonFile();
    setActivationType(data["activation"]);
    auto jLayers = data["layer"];
    auto numNeurons = jLayers.size();
    for (const auto &nLayer: jLayers) {
        layers.push_back(nLayer);
    }
    neuron.resize(numNeurons);
    weight.resize(numNeurons - 1);
    bias.resize(numNeurons);
    for (int i = 0; i < numNeurons; ++i) {
        neuron[i].resize(data["layer"][i]);
        bias[i].resize(data["layer"][i]);
    }
    for (int i = 0; i < weight.size(); ++i) {
        weight[i].resize(neuron[i + 1].size());
        for (int j = 0; j < weight[i].size(); ++j) {
            weight[i][j].resize(neuron[i].size());
        }
    }
    for (int i = 0; i < weight.size(); ++i) {
        for (int j = 0; j < weight[i].size(); ++j) {
            for (int k = 0; k < weight[i][j].size(); ++k) {
                weight[i][j][k] = data["weights"][i][j][k];
            }
        }
    }
    for (int i = 0; i < bias.size(); ++i) {
        for (int j = 0; j < bias[i].size(); ++j) {
            bias[i][j] = data["biases"][i][j];
        }

    }
    rawNeuron = neuron;
}

std::vector<double> NeuralNetwork::feed(const std::vector<double> &input) {
    if (input.size() != neuron[0].size()) {
        std::cerr << "Wrong input format" << std::endl;
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < input.size(); ++i) {
        neuron[0][i] = input[i];
        rawNeuron[0][i] = input[i];
    }
    feedForward();
    std::vector<double> result;
    for (double i: neuron[neuron.size() - 1]) {
        result.push_back((double) i);
    }
    return result;
}

void NeuralNetwork::feedForward() {
    for (int i = 1; i < neuron.size(); ++i) {
        for (int j = 0; j < neuron[i].size(); ++j) {
            neuron[i][j] = 0;
            rawNeuron[i][j] = 0;
        }
    }
    for (int i = 1; i < neuron.size(); ++i) {
        for (int j = 0; j < neuron[i].size(); ++j) {
            for (int k = 0; k < neuron[i - 1].size(); ++k) {
                rawNeuron[i][j] += weight[i - 1][j][k] * neuron[i - 1][k];
            }
            neuron[i][j] = activationFn(rawNeuron[i][j] + bias[i][j]);
        }
    }
}

void
NeuralNetwork::trainSubset(const TrainData &d,
                           std::vector<std::vector<std::vector<std::atomic<double>>>> &newWeights,
                           std::vector<std::vector<double>> relativeDeltaErrors,
                           std::atomic<double> &cost, double learningRate) {
    if (d.image.size() != neuron[0].size()) {
        std::cerr << "Wrong input format" << std::endl;
        exit(EXIT_FAILURE);
    }
    for (int j = 0; j < d.image.size(); ++j) {
        neuron[0][j] = d.image[j];
    }
    feedForward();
    cost.fetch_add(calculateCost(d.value));
    std::vector<double> target;
    target.resize(layers[layers.size() - 1]);
    for (int j = 0; j < target.size(); ++j) {
        if (j == d.value) target[j] = 1;
        else target[j] = 0;
    }
    backPropagate(target, std::move(relativeDeltaErrors), learningRate, newWeights);
}

void NeuralNetwork::train(const std::vector<std::vector<TrainData>> &data, unsigned int iterations,
                          double learningRate) {
    std::vector<std::vector<std::vector<double>>> minCostWeights;
    std::vector<std::vector<std::vector<std::atomic<double>>>> newWeights;
    std::vector<std::vector<double>> relativeDeltaErrors;
    relativeDeltaErrors.resize(layers.size());
    for (int j = 0; j < layers.size(); ++j) {
        relativeDeltaErrors[j].resize(layers[j], 0);
    }
    double minCost = -1;
    ThreadPool pool(6);
    for (int i = 0; i < iterations; ++i) {
        std::atomic<double> cost = 0;
        for (int l = 0; l < weight.size(); ++l) {
            for (int n = 0; n < weight[l].size(); ++n) {
                for (int w = 0; w < weight[l][n].size(); ++w) {
                    newWeights[l][n][w].store(weight[l][n][w]);
                }
            }
        }
        for (auto &set: data) {
            for (const auto &d: set) {
                TrainingTask t = {*this, d, newWeights, relativeDeltaErrors, cost, learningRate};
                pool.addTask(t);
            }
            pool.waitAllTasks();
            for (int l = 0; l < weight.size(); ++l) {
                for (int n = 0; n < weight[l].size(); ++n) {
                    for (int w = 0; w < weight[l][n].size(); ++w) {
                        weight[l][n][w] = newWeights[l][n][w].load();
                    }
                }
            }
        }
        double progress = (double) i * 100 / iterations;
        double totalCost = cost.load() / ((double) data.size() * (double) data[0].size());
        if (totalCost < minCost || minCost == -1) {
            minCost = totalCost;
            minCostWeights = weight;
        }
        std::cout << "\rProgress: " << std::fixed << std::setprecision(2) << progress << "% | " << "Total cost: "
                  << std::fixed << std::setprecision(8) << totalCost << std::flush;
        if (i % 1000 == 0) saveProgress();
    }
    updateJsonFile();
}


void NeuralNetwork::backPropagate(std::vector<double> target,
                                  std::vector<std::vector<double>> relativeDeltaErrors,
                                  double learningRate,
                                  std::vector<std::vector<std::vector<std::atomic<double>>>> &newWeights) {
    unsigned int lastLayerIndex = weight.size() - 1;
    unsigned int lastLayerNeuronsIndex = (neuron.size() - 1) - ((weight.size() - 1) - lastLayerIndex);
    /**
     * back propagate last layer
     *
     * using chain rule formula:
     * localCost = ∂Error/∂w = ∂Error/∂neuron * ∂neuron/∂rewNeuron * ∂neuron/∂weight
     * = 2*(output * target) * (rawNeuron)' * neuron
     *
     * newWeight -= localCost * learningRate
     *
     */
    for (int i = 0; i < weight[lastLayerIndex].size(); ++i) {
        double output = neuron[lastLayerNeuronsIndex][i];
        relativeDeltaErrors[lastLayerNeuronsIndex][i] = 2 * (output - target[i]);
        for (int j = 0; j < weight[lastLayerIndex][i].size(); ++j) {
            double source = neuron[lastLayerNeuronsIndex - 1][j];
            double rawSource = rawNeuron[lastLayerNeuronsIndex - 1][j];
            double localError =
                    relativeDeltaErrors[lastLayerNeuronsIndex][i] * activationFnDerivative(rawSource) * source;
            newWeights[lastLayerIndex][i][j].fetch_add(-learningRate * localError);
        }
    }

    /**
     * back propagate the rest
     *
     * because the weight in hidden layers are (from definition) connected to all neuron in the next layers
     * we need to first calculate the ∂Error/∂neuron = sum( ∂Error/∂next_neurons *  weight * raw_neuron )
     *
     * then the we can calculate the localCost using the very same formula as we used in backpropagation the last layer
     *
     */
    for (int i = (int) (lastLayerIndex - 1); i >= 0; --i) {
        for (int j = 0; j < weight[i].size(); ++j) {
            double subtotal = 0;
            for (int l = 0; l < relativeDeltaErrors[i + 2].size(); ++l) {
                subtotal += relativeDeltaErrors[i + 2][l] * activationFnDerivative(rawNeuron[i + 1][j]) *
                            weight[i + 1][l][j];
            }
            relativeDeltaErrors[i + 1][j] = subtotal;
            for (int k = 0; k < weight[i][j].size(); ++k) {
                double localError =
                        activationFnDerivative(rawNeuron[i + 1][j]) * neuron[i][k] * relativeDeltaErrors[i + 1][j];
                newWeights[i][j][k].fetch_add(-learningRate * localError);
            }
        }
    }
}

double NeuralNetwork::calculateCost(unsigned int targetValue) {
    double cost = 0;
    for (int i = 0; i < neuron[neuron.size() - 1].size(); ++i) {
        if (i == targetValue) {
            cost += pow((1 - (double) neuron[neuron.size() - 1][i]), 2);
        } else {
            cost += pow((0 - (double) neuron[neuron.size() - 1][i]), 2);
        }
    }
    return cost;
}

double NeuralNetwork::ReLU(double v) {
    return v > 0 ? v : 0;
}

double NeuralNetwork::ReLUDerivative(double v) {
    return v > 0 ? 1 : 0;
}

double NeuralNetwork::sigmoid(double v) {
    if (v >= 650) {
        return 1.0;
    } else if (v <= -650) {
        return 0.0;
    }
    return (1.0 / (1.0 + exp(-(double) v)));
}

double NeuralNetwork::sigmoidDerivative(double v) {
    double sig = sigmoid(v);
    double result = sig * (1.0 - sig);
    return result;
}

double NeuralNetwork::activationFn(double v) const {
    switch (activationType) {
        case 0:
            return sigmoid(v);
        case 1:
            return ReLU(v);
        default:
            std::cerr << "Invalid activation function" << std::endl;
            exit(EXIT_FAILURE);
    }
}

void NeuralNetwork::setActivationType(int v) {
    this->activationType = v;
}

double NeuralNetwork::activationFnDerivative(double v) const {
    switch (activationType) {
        case 0:
            return sigmoidDerivative(v);
        case 1:
            return ReLUDerivative(v);
        default:
            std::cerr << "Invalid activation function" << std::endl;
            exit(EXIT_FAILURE);
    }
}

void NeuralNetwork::initRandom() {
    for (auto &i: weight) {
        for (auto &j: i) {
            for (double &k: j) {
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_real_distribution<> dis(-2.0, 2.0);
                k = dis(gen);
            }
        }
    }
    for (auto &bia: bias) {
        for (double &j: bia) {
            /*std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> dis(-1.0, 1.0);
            j = dis(gen);*/
            j = 0;
        }
    }
    for (double &i: bias[bias.size() - 1]) {
        i = 0;
    }
}

void NeuralNetwork::initJsonFile() {
    std::string file;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1.0);
    file +=
            "neuralNetwork(layers=" + std::to_string(neuron.size()) + ", id=" +
            std::to_string((int) round(dis(gen) * 10000)) +
            ").json";
    std::ofstream output_file(file);
    filename = file;
    writeJsonFile(weight, bias, true);
}

void NeuralNetwork::saveProgress() {
    writeJsonFile(weight, bias, false);
}

void NeuralNetwork::updateJsonFile() {
    writeJsonFile(weight, bias, true);
}

void NeuralNetwork::writeJsonFile(std::vector<std::vector<std::vector<double>>> &weightToSave,
                                  std::vector<std::vector<double>> &biasToSave, bool print) {
    std::ofstream output_file(filename);
    if (output_file.is_open()) {
        json data;
        data["activation"] = activationType;
        data["layer"] = layers;
        data["weights"] = weightToSave;
        data["biases"] = biasToSave;
        std::string stringData = data.dump(4);
        output_file << data.dump(4);
        output_file.close();
        if (print) std::cout << "JSON data saved to '" << filename << "'" << std::endl;
    } else {
        std::cerr << "Failed to open the output file." << std::endl;
    }
}

nlohmann::json NeuralNetwork::readJsonFile() {
    std::ifstream jFile(filename);
    if (!jFile.is_open()) {
        std::cerr << "Failed to open the JSON file." << std::endl;
        exit(EXIT_FAILURE);
    }
    std::string jsonString((std::istreambuf_iterator<char>(jFile)), std::istreambuf_iterator<char>());
    jFile.close();
    return json::parse(jsonString);
}

std::string NeuralNetwork::toString() {
    std::string result;
    result += "neurons: \n";
    for (auto &i: neuron) {
        for (double j: i) {
            result += std::to_string(j);
            result += " ";
        }
        result += "\n";
    }
    result += "weights: \n";
    for (auto &i: weight) {
        for (auto &j: i) {
            for (double k: j) {
                result += std::to_string(k);
                result += " ";
            }
            result += "\n";
        }
        result += "\n\n";
    }
    return result;
}
