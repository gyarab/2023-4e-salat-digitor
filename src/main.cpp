#include <iostream>
#include "Layer.h"
#include "NeuralNetwork.h"
#include "random"

using namespace std;


int main() {
    Layer input = Layer(10);
    for (int i = 0; i < 10; i++) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0, 256);
        input.neurons[i]->setValue(dis(gen));
    }
    NeuralNetwork digitor = NeuralNetwork({input, Layer(5), Layer(5), Layer(10)});
    cout << "Hello, digitor" << endl;
    digitor.forwardProp();
    vector<Neuron *> outputL = digitor.layers[digitor.layers.size() - 1].neurons;
    cout << outputL.size() << endl;
    for (auto &i: outputL) {
        cout << "Outputs: " + to_string(i->getValue()) << endl;
    }
    return 0;
}

