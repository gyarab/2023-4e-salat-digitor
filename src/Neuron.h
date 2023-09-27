//
// Created by ondrej on 26/09/23.
//

#ifndef DIGITOR_NEURON_H
#define DIGITOR_NEURON_H

#include "string"

class Neuron {
public:
    Neuron();

    void addValue(double v);

    void setBias(double b);

    void setValue(double v);

    [[nodiscard]] double getOutput() const;

    [[nodiscard]] double getValue() const;

    [[nodiscard]] static double sigmoid(double d);

    [[nodiscard]] double getBias() const;

    [[nodiscard]] std::string toString() const;

    double value{};
    double bias{};
private:


};


#endif //DIGITOR_NEURON_H
