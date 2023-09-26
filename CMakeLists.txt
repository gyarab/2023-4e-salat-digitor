cmake_minimum_required(VERSION 3.26)
project(digitor)

set(CMAKE_CXX_STANDARD 23)

add_executable(digitor src/main.cpp
        src/NeuralNetwork.cpp
        src/NeuralNetwork.h
        src/Neuron.cpp
        src/Neuron.h
        src/Connection.cpp
        src/Connection.h
        src/Layer.cpp
        src/Layer.h)
