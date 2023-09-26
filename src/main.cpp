#include <iostream>
#include "Layer.h"
#include "NeuralNetwork.h"

using namespace std;

int main() {
    Layer input = Layer(10);
    Layer hidden1 = Layer(5);
    Layer hidden2 = Layer(5);
    Layer output = Layer(10);
    NeuralNetwork digitor = NeuralNetwork({input, hidden1, hidden2, output});
    cout << "Hello, digitor" << endl;
    cout << digitor.toString() << endl;
    return 0;
}

