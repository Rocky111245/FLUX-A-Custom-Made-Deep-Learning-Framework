#ifndef DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_NEURAL_NETWORK_FRAMEWORK_H
#define DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_NEURAL_NETWORK_FRAMEWORK_H
#include "MatrixLibrary.h"
#include <iostream>
#include <initializer_list>
#include <vector>
#include <cmath>
#include <cstring>
#include <memory> // Include for unique_ptr


enum class ActivationType {
    RELU,
    SIGMOID,
    TANH,
    LEAKY_RELU,
    SWISH,
    LINEAR
};

class Neural_Layer {
private:
    int neurons_in_first_layer, neurons_in_second_layer;

public:
    Matrix weights_matrix; // holds the weights of every connection within a layer
    Matrix bias_matrix; //holds the bias of every connection within a layer
    Matrix input_matrix; //holds the input values of every connection within a layer
    Matrix weight_and_input_matrix; // after weight is linearly combined with the input
    Matrix weight_input_bias_matrix; //after weight,bias and input is linearly combined
    Matrix activated_output_matrix; // this holds the activated output of that layer
    std::unique_ptr<Matrix> dC_dy_matrix; // derivative of cost function with respect to experimental output
    Matrix dC_da_matrix; // derivative of cost function with respect to the inputs
    Matrix dC_dw_matrix; //derivative of the cost function with respect to the weights in every layer
    Matrix dC_db_matrix; //derivative of the cost function with respect to the bias in every layer
    Matrix dh_da_matrix; // derivative of the activation function with respect to its inputs
    std::unique_ptr<Matrix> C; // a matrix to hold the cost function, this makes calculation easier
    struct activation_functions{
        ActivationType hidden_layers_activation_function;
        ActivationType last_layer_activation_function;
    } layer_activation;

    // Constructors
    Neural_Layer();
    Neural_Layer(int neuronsInFirstLayer, int neuronsInSecondLayer, const Matrix& inputMatrix, ActivationType hidden_activation, ActivationType output_activation);

    // Member functions
    void Compute_Weighted_Sum();
    void Activate(ActivationType activation_type) ;
    void Dh_Da_Function(bool is_last_layer) ;
    void Initialize_dC_dy_Matrix();
    void Initialize_Cost_Function_Matrix();
    Matrix Initialize_Weights(int row, int column);


    // Static member functions
    static float Sigmoid_Function(float x);
    static float ReLU(float x);
    static float LeakyReLU(float x);
    static float ELU(float x, float alpha);
    static float Swish(float x);
    static float Tanh(float x);
    static float Linear_Activation(float x);
};

struct Neural_Layer_Information {
    std::vector<Neural_Layer> neural_layers;
    std::vector<int> layers_vector;
    int sample_size;
    Matrix outputMatrix;
};

Neural_Layer_Information Form_Network(std::initializer_list<int> layers, Matrix inputMatrix, const Matrix& outputMatrix, ActivationType hidden_activation, ActivationType output_activation);
static void displayLayerDetails(const Neural_Layer_Information &layers);
void Forward_Pass(Neural_Layer_Information &neural_layer_information);
void Back_Propagation(Neural_Layer_Information &neural_layer_information, float &mean_squared_error);
void Learn(Neural_Layer_Information &neural_layer_information, float learning_rate, int iterations);
void Matrix_Fill(Matrix& matrix, float value);
Matrix BinaryThreshold(const Matrix& input, float threshold = 0.5f);

#endif //DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_NEURAL_NETWORK_FRAMEWORK_H
