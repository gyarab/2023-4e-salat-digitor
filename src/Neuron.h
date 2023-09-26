//
// Created by ondrej on 26/09/23.
//

#ifndef DIGITOR_NEURON_H
#define DIGITOR_NEURON_H
#include "string"

class Neuron {
public:
    Neuron();

    void addValue(double value);

    [[nodiscard]] double getOutput() const;

    [[nodiscard]] double getBias() const;

    void setBias(double bias);
    std::string toString() const;
private:
    double value;
    double bias;

};


#endif //DIGITOR_NEURON_H
