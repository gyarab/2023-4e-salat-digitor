#include "NeuralNetwork.h"
#include "iostream"
#include "TrainData.h"


using namespace std;

static void printVector(vector<double> v) {
    cout << "[";
    for (int i = 0; i < v.size(); ++i) {
        if (i == 0) cout << v[i];
        else cout << ", " << v[i];
    }
    cout << "]" << endl;
}

int main() {
    vector<unsigned int> layers = {5, 2, 2, 4};
    vector<double> input = {1, 2, 3, 4, 5};
    vector<TrainData> trainData = {{{1, 2, 3, 4, 5},  1},
                                   {{2, 3, 4, 5, 6}, 2}};
    //NeuralNetwork digitor = NeuralNetwork(layers, "sigmoid");
    //digitor.feed(input);
    NeuralNetwork d = NeuralNetwork("neuralNetwork(layers=4, id=9039).json");
    vector<double> output = d.feed(input);
    printVector(output);
    d.train(trainData);
    return 0;
}




