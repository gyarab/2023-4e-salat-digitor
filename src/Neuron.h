//
// Created by ondrej on 26/09/23.
//

#ifndef DIGITOR_NEURON_H
#define DIGITOR_NEURON_H


class Neuron {
public:
    Neuron();

    void addValue(double value);

    [[nodiscard]] double getOutput() const;

    [[nodiscard]] double getBias() const;

    void setBias(double bias);

private:
    double value;
    double bias;

};


#endif //DIGITOR_NEURON_H
