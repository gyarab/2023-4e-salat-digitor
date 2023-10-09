#include <iostream>
#include "NeuralNetwork.h"

using namespace std;


int main() {
    vector<int> layers = {20, 5, 5, 9};
    NeuralNetwork digitor = NeuralNetwork(layers);
    vector<double> input = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    digitor.feed(input);
    return 0;
}

