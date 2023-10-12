#include <iostream>
#include "NeuralNetwork.h"

using namespace std;


int main() {
    vector<int> layers = {5, 2, 2, 4};
    NeuralNetwork digitor = NeuralNetwork(layers);
    vector<double> input = {1, 2, 3, 4, 5};
    //digitor.feed(input);
    NeuralNetwork d = NeuralNetwork("neuralNetwork(layers=4, id=8978).json");
    d.feed(input);
    return 0;
}

