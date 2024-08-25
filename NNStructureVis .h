#ifndef _DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_NNSTRUCTUREVIS_H
#define _DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_NNSTRUCTUREVIS_H

#include "Neural Network Framework.h"
#include <string>

class NNStructureVis {
public:
    static void visualizeNetwork(const Neural_Layer_Information& network);

private:
    static std::string getLayerString(int neurons, const std::string& label);
    static std::string getConnectionString(int inputNeurons, int outputNeurons);
    static std::string centerString(const std::string& s, int width);
};





#endif //_DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_NNSTRUCTUREVIS_H
