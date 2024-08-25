#include <vector>
#include "Neural Network Framework.h"
#include "MSEGraphPlotter.h"
#include "NNStructureVis .h"



int main() {
    // 4-bit Parity Test
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


// first 4 columns are the input. the last column is the output. The goal is to see if my neural network framework can model this non-linear input.


    Matrix input_matrix = Matrix_Data_Preprocessor(16, 4, 0, 0, inputData);
    //Matrix::Print(input_matrix);
    Matrix output_matrix = Matrix_Data_Preprocessor(16, 1, 4, 0, inputData);
    //Matrix::Print(output_matrix);


    // Create a neural network of 4 input neurones (implicit) and a hidden layer of 8 neurones,followed by 4 and output neurone of 1
    auto neural_layer_information = Form_Network({8,4, 1}, input_matrix, output_matrix, ActivationType::LEAKY_RELU, ActivationType::SIGMOID);

    // Visualize the neural network structure
    NNStructureVis::visualizeNetwork(neural_layer_information);

    // Train the network (this will later the MSE graph at the end)
    // learning rate is 0.01 and iteration is 350000
    Learn(neural_layer_information, 0.01, 350000);

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