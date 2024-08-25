#include "NNStructureVis .h"
#include <iostream>
#include <sstream>
#include <algorithm>

void NNStructureVis::visualizeNetwork(const Neural_Layer_Information& network) {
    const int width = 80;
    std::cout << std::string(width, '-') << std::endl;
    std::cout << centerString("Neural Network Structure", width) << std::endl;
    std::cout << std::string(width, '-') << std::endl;

    // Input layer (based on the number of columns in the input matrix)
    int inputNeurons = network.neural_layers[0].input_matrix.columns();
    std::cout << getLayerString(inputNeurons, "Input") << std::endl;
    std::cout << getConnectionString(inputNeurons, network.layers_vector[0]) << std::endl;

    // Hidden and output layers
    for (size_t i = 0; i < network.layers_vector.size(); ++i) {
        std::string label = (i == network.layers_vector.size() - 1) ? "Output" : "Hidden";
        std::cout << getLayerString(network.layers_vector[i], label) << std::endl;

        if (i < network.layers_vector.size() - 1) {
            std::cout << getConnectionString(network.layers_vector[i], network.layers_vector[i+1]) << std::endl;
        }
    }

    std::cout << std::string(width, '-') << std::endl;
    std::cout << "Legend: (o) - Neuron, | - Connection" << std::endl;
    std::cout << std::string(width, '-') << std::endl;
}

std::string NNStructureVis::getLayerString(int neurons, const std::string& label) {
    std::stringstream ss;
    ss << label << " Layer: ";
    for (int i = 0; i < neurons; ++i) {
        ss << "(o) ";
    }
    ss << "(" << neurons << " neurons)";
    return ss.str();
}

std::string NNStructureVis::getConnectionString(int inputNeurons, int outputNeurons) {
    std::stringstream ss;
    ss << std::string(13, ' ');
    int connections = std::min(inputNeurons, outputNeurons);
    for (int i = 0; i < connections; ++i) {
        ss << "| ";
    }
    return ss.str();
}

std::string NNStructureVis::centerString(const std::string& s, int width) {
    int padding = width - s.length();
    int padLeft = padding / 2;
    int padRight = padding - padLeft;
    return std::string(padLeft, ' ') + s + std::string(padRight, ' ');
}