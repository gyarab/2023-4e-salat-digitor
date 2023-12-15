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
    vector<unsigned int> layers = {784, 16, 16, 10};
    NeuralNetwork d = NeuralNetwork("neuralNetwork(layers=4, id=2753).json");
    vector<TrainData> train;
    for (int j = 0; j < 10; j++) {
        vector<double> input;
        input.resize(784);
        for (int i = 0; i < 784; ++i) {
            cin >> input[i];
        }
        int target;
        cin >> target;
        train.push_back({input, static_cast<unsigned int>(target)});
    }
    d.train(train, 1000, 0.3);
    //vector<double> input = {1, 2, 3, 4, 5};
    //vector<TrainData> trainData = {{{1, 2, 3, 4, 5}, 1},};
    //NeuralNetwork digitor = NeuralNetwork(layers, "sigmoid");
    //digitor.feed(input);
    //printVector(output);
    //d.train(trainData, 100, 0.6);
    //output = d.feed(input);
    //printVector(output);
    return 0;
}




