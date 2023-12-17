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
    NeuralNetwork d = NeuralNetwork("neuralNetwork(layers=4, id=3308).json");
    /*vector<double> input;
    input.resize(784);
    for (int i = 0; i < 784; ++i) {
        cin >> input[i];
    }
    printVector(d.feed(input));*/
    vector<TrainData> train;
    for (int j = 0; j < 1000; j++) {
        vector<double> input;
        input.resize(784);
        for (int i = 0; i < 784; ++i) {
            cin >> input[i];
        }
        int target;
        cin >> target;
        train.push_back({input, static_cast<unsigned int>(target)});
    }
    d.train(train, 1000, 0.6);
    /*vector<unsigned int> layers = {5, 2, 2, 4};
    vector<double> input = {1, 2, 3, 4, 5};
    vector<TrainData> trainData = {{{1, 2, 3, 4, 5}, 1},};
    NeuralNetwork digitor = NeuralNetwork("neuralNetwork(layers=4, id=2616).json");
    vector<double> output = digitor.feed(input);
    printVector(output);
    digitor.train(trainData, 1000, 0.6);
    output = digitor.feed(input);
    printVector(output);*/
    return 0;
}




