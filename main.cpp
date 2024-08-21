#include <vector>
#include "Neural Network Framework.h"




int main() {
    // 3D XOR gate input and output data
    std::vector<std::vector<float>> inputData = {
            {0, 0, 0, 0, 0},
            {0, 0, 0, 1, 1},
            {0, 0, 1, 0, 1},
            {0, 0, 1, 1, 0},
            {0, 1, 0, 0, 1},
            {0, 1, 0, 1, 0},
            {0, 1, 1, 0, 0},
            {0, 1, 1, 1, 1},
            {1, 0, 0, 0, 1},
            {1, 0, 0, 1, 0},
            {1, 0, 1, 0, 0},
            {1, 0, 1, 1, 1},
            {1, 1, 0, 0, 0},
            {1, 1, 0, 1, 1},
            {1, 1, 1, 0, 1},
            {1, 1, 1, 1, 0}
    };


    Matrix input_matrix = Matrix_Data_Preprocessor(16, 4, 0, 0, inputData);
    Matrix::Print(input_matrix);
    Matrix output_matrix = Matrix_Data_Preprocessor(16, 1, 4, 0, inputData);
    Matrix::Print(output_matrix);

    // Create a neural network with 16 input neurons(inbuilt), 8 hidden neurons, and 1 output neuron
    auto neural_layer_information = Form_Network({8,4, 1}, input_matrix, output_matrix, ActivationType::TANH, ActivationType::SIGMOID);

    Learn(neural_layer_information, 0.01, 300000);

    return 0;
}


//Under Test environment


//    Matrix inputs= Matrix_Maker_2DArray(4, 8, 3, 0, &data[0][0]);
//    Matrix_Display(inputs);
//
//    Matrix output= Matrix_Maker_2DArray(4,8,1,3,&data[0][0]);
//    Matrix_Display(output);
//
//

//    std::vector<Neural_Layer> network = Form_Network({3,3,1}, inputs);
//    Learn(network,{3,3,1},inputs,output,0.0014,405000);