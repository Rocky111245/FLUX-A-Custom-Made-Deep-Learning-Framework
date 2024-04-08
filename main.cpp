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
    Matrix da_dw_matrix;
    Matrix dC_da_matrix;
    Matrix dh_da_matrix;
    Matrix da_dh_matrix;



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
              da_dw_matrix(Matrix_Create_Zero(neuronsInSecondLayer, neuronsInFirstLayer)),
              dC_da_matrix(Matrix_Create_Zero(neuronsInSecondLayer, 1)),
              dh_da_matrix(Matrix_Create_Zero(neuronsInSecondLayer, 1)),
              da_dh_matrix(Matrix_Create_Zero(neuronsInSecondLayer, neuronsInFirstLayer))
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

    void Da_Dw_Function(){
        da_dw_matrix=input_matrix;
    }
    void Da_Dh_Function(){
        da_dh_matrix=weights_matrix;
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

    void Display_Weight_Derivatives() const{
        Matrix_Display(da_dw_matrix);
    }
    void Display_Activated_Function_Derivatives() const{
        Matrix_Display(dh_da_matrix);
    }
    void Display_Activated_Function_Derivatives_Respect_To_Input() const{
        Matrix_Display(da_dh_matrix);
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
    Matrix_Display(output);

    forwardPass({4,3,2,1},inputs);
}

#include <cmath> // For std::exp







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
        neural_layers[i].Da_Dw_Function();
        neural_layers[i].Dh_Da_Function();
        neural_layers[i].Da_Dh_Function();
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
        std::cout<<"WEIGHT DERIVATIVES"<<std::endl;
        neural_layers[i].Display_Weight_Derivatives();
        std::cout<<"ACTIVATED DERIVATIVES"<<std::endl;
        neural_layers[i].Display_Activated_Function_Derivatives();
        std::cout<<"ACTIVATED DERIVATIVES WITH RESPECT TO INPUTS"<<std::endl;
        neural_layers[i].Display_Activated_Function_Derivatives();
        std::cout<<"0 matrix"<<std::endl;
        Matrix_Display(neural_layers[i].dC_da_matrix);
        neural_layers[i].Display_Activated_Function_Derivatives_Respect_To_Input();



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
     neural_layers[size-2].dC_dY = (float)((2 / 5) * (neural_layers[size - 2].weight_input_bias_matrix.data[0] - 1));

    return neural_layers;
}

//suppose a fully connected dense neural network
void backPropagation( const std::vector<Neural_Layer>&neural_layers,std::initializer_list<int> layers){

    // Convert initializer_list to vector for easier access
    std::vector<int> neuronesInCurrentLayer = layers;
    size_t size = neuronesInCurrentLayer.size();
    std::cout<<"Size of Neural Layer: "<<size<<std::endl;

    //we need to get this dynamically, this is for testing
    float dC_dY=neural_layers[size-2].dC_dY;
    int neurone_in_front_layer;
    neural_layers[size-2].dC_da_matrix.data[0]= dC_dY * 1;//multiplied by 1 since dY_dA is 1 since no activation

    //this for loop takes care of the layer number
    for (int layer_number= (int)(size - 2); layer_number > 0; layer_number--){
        //this for loop will iterate as many times as there are Neurones
        int i=(int)size-2;
        for (int count=0; count <neuronesInCurrentLayer[i];count++ ){
            //this for loop is the gradient addition for each neurone itself
            int j=(int)size-3;
            for (int neurone_number=neuronesInCurrentLayer[j+1];neurone_number>=0;neurone_number--){
                

            }

        }

    }
}

float calculateCost_Function_Respect_to_backOutput(Matrix A,int size){


}




