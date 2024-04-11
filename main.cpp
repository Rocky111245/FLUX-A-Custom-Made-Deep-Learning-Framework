#include <iostream>
#include <initializer_list>
#include <vector>
#include <cmath>

extern "C" {
#include "library.h"
}

class Neural_Layer {
private:
    int neurons_in_first_layer, neurons_in_second_layer;

public:
    float dC_dY;

    Matrix weights_matrix;
    Matrix bias_matrix;
    Matrix input_matrix;
    Matrix actual_output_matrix;
    Matrix weight_and_input_matrix;
    Matrix weight_input_bias_matrix;
    Matrix activated_output_matrix;
    Matrix dC_da_matrix;
    Matrix dC_dw_matrix;
    Matrix dh_da_matrix;


    Neural_Layer() {
        // Default constructor body (can be empty)
    }
    // Parameterized constructor
    Neural_Layer(int neuronsInFirstLayer, int neuronsInSecondLayer, const Matrix& inputMatrix)
            : neurons_in_first_layer(neuronsInFirstLayer),
              neurons_in_second_layer(neuronsInSecondLayer),
              dC_dY(0),
              input_matrix(inputMatrix),
              weights_matrix(Matrix_Create_Random(neuronsInSecondLayer, neuronsInFirstLayer, 2)),
              bias_matrix(Matrix_Create_Random(neuronsInSecondLayer, 1, 2)),
              weight_and_input_matrix(Matrix_Create_Zero(neuronsInSecondLayer, 1)),
              weight_input_bias_matrix(Matrix_Create_Zero(neuronsInSecondLayer, 1)),
              activated_output_matrix(Matrix_Create_Zero(neuronsInSecondLayer, 1)),
              dC_da_matrix(Matrix_Create_Zero(neuronsInSecondLayer, 1)),
              dh_da_matrix(Matrix_Create_Zero(neuronsInSecondLayer, 1)),
              dC_dw_matrix(Matrix_Create_Zero(neuronsInSecondLayer,neuronsInFirstLayer))


    {}

    void Compute_Weighted_Sum() {
        Matrix_Multiply(&weight_and_input_matrix, weights_matrix, input_matrix);
        Matrix_Add(&weight_input_bias_matrix, weight_and_input_matrix, bias_matrix);
    }

    void Activate() {
        for (int i = 0; i < weight_input_bias_matrix.row; ++i) {
            activated_output_matrix.data[i] = ReLU(weight_input_bias_matrix.data[i]);

        }
    }
    void Activate_Last(){
        for (int i = 0; i < weight_input_bias_matrix.row; ++i) {
            activated_output_matrix.data[i] = Sigmoid_Function(weight_input_bias_matrix.data[i]);
        }
    }
    void Dh_Da_Function(){
        for (int i = 0; i < neurons_in_second_layer; i++){
            dh_da_matrix.data[i] = (weight_input_bias_matrix.data[i] > 0) ? 1.0f : 0.0f;
        }
    }

    void Display_Inputs() const{
        Matrix_Display(input_matrix);
    }
    void Display_Weights() const{
        Matrix_Display(weights_matrix);

    }
    void Display_Weighted_Sum() const {
        Matrix_Display(weight_and_input_matrix);
    }
    void Display_Bias_Weighted_Sum() const {
        Matrix_Display(weight_input_bias_matrix);
    }

    void Display_Bias() const {
        Matrix_Display(bias_matrix);
    }

    void Display_Activated_Output() const {
        Matrix_Display(activated_output_matrix);
    }

    void Display_Activated_Function_Derivatives() const{
        Matrix_Display(dh_da_matrix);
    }


private:
    static float Sigmoid_Function(float x) {
        return 1.0f / (1.0f + std::exp(-x));
    }
    // ReLU activation function
    static float ReLU(float x) {
        return std::max(0.0f, x);
    }

// Leaky ReLU activation function
    static float LeakyReLU(float x, float alpha) {
        return (x > 0) ? x : alpha * x;
    }

// ELU activation function
    static float ELU(float x, float alpha) {
        return (x > 0) ? x : alpha * (std::exp(x) - 1);
    }

// Swish activation function
    static float Swish(float x) {
        return x * (1.0f / (1.0f + std::exp(-x)));
    }

// Tanh activation function
    static float Tanh(float x) {
        return std::tanh(x);
    }
};


Matrix* Neural_Layer_Maker(int neurones_In_First_Layer, int neurones_In_Second_Layer,Matrix inputMatrix);
std::vector<Neural_Layer> forwardPass(std::initializer_list<int> layers, Matrix inputMatrix);
void backPropagation( std::vector<Neural_Layer>&neural_layers,std::initializer_list<int> layers);
void Matrix_Transpose_v2(Matrix *final, const Matrix original) ;
void Matrix_Multiply_V2(Matrix *finalMatrix, Matrix firstMatrix, Matrix secondMatrix);

int main(){
    // Seed the random number generator
    srand((unsigned) time(NULL));
    float data[][3]={
            {1,0,1},
            {0,1,0},
            {1,1,1},
            {0,0,0}
    };
    size_t sizeOfData=sizeof(data)/sizeof(data[0]);

    Matrix inputs= Matrix_Maker_2DArray(3, 4, 1, 0, &data[0][0]);
    Matrix output= Matrix_Maker_2DArray(3,4,1,2,&data[0][0]);
    std::vector g=forwardPass({4,3,2,1},inputs);
    backPropagation(g,{4,3,2,1});
}









//this function will iterate the Neural_Layer_Maker function on the number of networks I want to make

std::vector<Neural_Layer> forwardPass(std::initializer_list<int> layers, Matrix inputMatrix)
{
    // Convert initializer_list to vector for easier access
    std::vector<int> layerSizes = layers;
    size_t size = layerSizes.size();
    std::cout<<"Size: "<<size<<std::endl;

    std::vector<Neural_Layer> neural_layers(size - 1);//layers will always be n-1 because the input layer is not considered
    // Initialize input matrix for the first layer
    Matrix currentInput = inputMatrix;

    // Iterate over each layer
    for (size_t i = 0; i < size - 1; i++) {
        // Create a neural layer with appropriate size and input
        neural_layers[i]= Neural_Layer(layerSizes[i], layerSizes[i + 1], currentInput);
        printf("Size: %zu\n", neural_layers.size()); // Print the size of the vectorNeural_Node.size();
        printf("Capacity: %zu\n", neural_layers.capacity()); // Print the capacity of the vectorNeural_Node.capacity();
        std::cout<<"--------------"<<std::endl;
        // Compute weighted sum and Activate the layer
        neural_layers[i].Compute_Weighted_Sum();
        neural_layers[i].Activate();
        neural_layers[i].Dh_Da_Function();
        // Display outputs for debugging or analysis
        std::cout<<"Layer"<<i+1<<std::endl;
        std::cout<<std::endl;
        std::cout<<"INPUTS"<<std::endl;
        neural_layers[i].Display_Inputs();
        std::cout<<"WEIGHTS"<<std::endl;
        neural_layers[i].Display_Weights();
        std::cout<<"WEIGHTS+INPUTS"<<std::endl;
        neural_layers[i].Display_Weighted_Sum();
        std::cout<<"WEIGHTS+INPUTS+BIAS"<<std::endl;
        neural_layers[i].Display_Bias_Weighted_Sum();
        std::cout<<"BIAS"<<std::endl;
        neural_layers[i].Display_Bias();
        std::cout<<"dC_da_matrix"<<std::endl;
        Matrix_Display(neural_layers[i].dC_da_matrix);
        std::cout<<"dC_dw_matrix"<<std::endl;
        Matrix_Display(neural_layers[i].dC_dw_matrix);
        std::cout<<"dh_da_matrix"<<std::endl;
        Matrix_Display(neural_layers[i].dh_da_matrix);
//        std::cout<<"dC_dw_matrix"<<std::endl;
//        Matrix_Display(neural_layers[i].dC_dw_matrix);
        std::cout<<"---------END OF LAYER--------------------"<<std::endl;

        if(i==size-2){
            neural_layers[i].Activate_Last();
        }
        else{
            neural_layers[i].Activate();
        }
        neural_layers[i].Display_Activated_Output();
        std::cout<<"--------------"<<std::endl;


        // Set the output of this layer as the input for the next layer
        currentInput = neural_layers[i].activated_output_matrix;
    }

    neural_layers[size-2].dC_dY = (2.0f / 5.0f) * (neural_layers[size - 2].weight_input_bias_matrix.data[0] - 1);

    return neural_layers;
}


void Matrix_Transpose_v2(Matrix *final, const Matrix original) {
    // This version assumes:
    // 1. 'final' has been allocated with enough memory to hold the transposed matrix.
    // 2. 'final' dimensions (row and column) have been set to match the transposed dimensions of 'original'.

    for (int i = 0; i < original.row; i++) {
        for (int j = 0; j < original.column; j++) {
            // Calculate the index in the original matrix
            int originalMatrixIndex = i * original.column + j;
            // Calculate the index in the transposed (final) matrix
            // Note: Assuming 'final' has its dimensions set to the transposed dimensions outside this function
            int finalMatrixIndex = j * final->column + i;
            // Assign the transposed value
            final->data[finalMatrixIndex] = original.data[originalMatrixIndex];
        }
    }
}







//suppose a fully connected dense neural network
void backPropagation( std::vector<Neural_Layer>&neural_layers,std::initializer_list<int> layers){

    // Convert initializer_list to vector for easier access
    // Assuming 'layers' is your std::initializer_list<int>
    std::vector<int> neuronesInCurrentLayer(layers.begin() + 1, layers.end());
    size_t size = neuronesInCurrentLayer.size();
    std::cout<<"Size of Neural Layer: "<<size<<std::endl;// This equals 3 3

    //we need to get this dynamically, this is for testing
    float dC_dY=neural_layers[size-1].dC_dY; //last layer
    std::cout<<"float value: "<<neural_layers[size-1].dC_dY<<std::endl;

    int neurone_in_front_layer;
    neural_layers[size-1].dC_da_matrix.data[0]= dC_dY * 1;//multiplied by 1 since dY_dA's been 1 since no activation
    int i=(int)size-1;  //last layer-->2
    int j=(int)size-2;  //before last layer-->1
    std::cout<<neuronesInCurrentLayer[0]<<std::endl;
    std::cout<<neuronesInCurrentLayer[1]<<std::endl;
    std::cout<<neuronesInCurrentLayer[2]<<std::endl;
    Matrix_Display(neural_layers[i].dC_da_matrix);



    for (int layer_number=i;layer_number>=0;layer_number--)
    {
        std::cout << "---------------------------" << std::endl;
        std::cout << "Layer Number: " << layer_number << std::endl;
        std::cout << "---------------------------" <<  std::endl;

        std::cout << "dC_da Matrix--Previous Layer before Operation 1-TARGETTED: " << std::endl;
        Matrix_Display(neural_layers[layer_number - 1].dC_da_matrix);
        std::cout << "dC_da Matrix--Current Layer involved in Operation 1: " << std::endl;
        Matrix_Display(neural_layers[layer_number].dC_da_matrix);
        std::cout << "Weights Matrix--Current Layer involved in Operation 1: " << std::endl;
        Matrix_Display(neural_layers[layer_number].weights_matrix);

        std::cout << "------TRANSPOSE OPERATION 1 STARTED-------- " << std::endl;
        Matrix n = Matrix_Create_Zero(neural_layers[layer_number].dC_da_matrix.column, neural_layers[layer_number].dC_da_matrix.row);
        Matrix_Transpose_v2(&n, neural_layers[layer_number].dC_da_matrix);
        std::cout << "------TRANSPOSE OPERATION 1 ENDED-------- " << std::endl;
        std::cout << "dC_da Matrix--Current Layer after Operation 1 (Transposed): " << std::endl;
        Matrix_Display(n);

        if (layer_number != 0) {
            std::cout << "------OPERATION 1 STARTED-MULTIPLY-------- " << std::endl;
            Matrix_Multiply(&neural_layers[layer_number - 1].dC_da_matrix, n, neural_layers[layer_number].weights_matrix);
            Matrix_Free(&n);
            std::cout << "------OPERATION 1 ENDED-MULTIPLY-------- " << std::endl;
        }




        std::cout << "dC_da Matrix--Previous Layer after Operation 1 TARGETTED: " << std::endl;
        Matrix_Display(neural_layers[layer_number - 1].dC_da_matrix);

        std::cout << "Input Matrix--Current Layer involved in Transpose Operation 1: " << std::endl;
        Matrix_Display(neural_layers[layer_number].input_matrix);
        std::cout << "------TRANSPOSE OPERATION 2 STARTED-MULTIPLY-------- " << std::endl;
        Matrix m = Matrix_Create_Zero(neural_layers[layer_number].input_matrix.column, neural_layers[layer_number].input_matrix.row);
        Matrix_Transpose_v2(&m, neural_layers[layer_number].input_matrix);
        std::cout << "------TRANSPOSE OPERATION 2 ENDED-------- " << std::endl;



        std::cout << "After Transpose: Matrix M " << std::endl;
        Matrix_Display(m);


        std::cout << "Current Layer Global Derivative dC/dW before Operation 2-TARGETTED" << std::endl;
        Matrix_Display(neural_layers[layer_number].dC_dw_matrix);
        std::cout << "Current Layer dC_da involved in Operation 2" << std::endl;
        Matrix_Display(neural_layers[layer_number].dC_da_matrix);

        std::cout << "------ OPERATION 2 STARTED (MULTIPLY) -------- " << std::endl;
        Matrix_Multiply(&neural_layers[layer_number].dC_dw_matrix, neural_layers[layer_number].dC_da_matrix, m);
        Matrix_Free(&m);

        std::cout << "------ OPERATION 2 ENDED (MULTIPLY) -------- " << std::endl;

        std::cout << "Current Layer Global Derivative dC/dW after Operation 2-TARGETTED" << std::endl;
        Matrix_Display(neural_layers[layer_number].dC_dw_matrix);
    }

}
