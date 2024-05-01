
#ifndef DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_NEURAL_NETWORK_FRAMEWORK_H
#define DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_NEURAL_NETWORK_FRAMEWORK_H
#include <iostream>
#include <initializer_list>
#include <vector>
#include <cmath>
#include <cstring>
extern "C" {
#include "library.h"
}


class Neural_Layer {
private:
    int neurons_in_first_layer, neurons_in_second_layer;

public:
    Matrix weights_matrix;
    Matrix bias_matrix;
    Matrix input_matrix;
    Matrix weight_and_input_matrix;
    Matrix weight_input_bias_matrix;
    Matrix activated_output_matrix;
    Matrix dC_dy_matrix;
    Matrix dC_da_matrix;
    Matrix dC_dw_matrix;
    Matrix dC_db_matrix;
    Matrix dh_da_matrix;
    Matrix C;


    // Constructors
    Neural_Layer();
    Neural_Layer(int neuronsInFirstLayer, int neuronsInSecondLayer, const Matrix& inputMatrix);

    // Member functions
    void Compute_Weighted_Sum();
    void Activate();
    void Activate_Last() ;
    void Dh_Da_Function();
    void Initialize_dC_dy_Matrix();
    void Initialize_Cost_Function_Matrix();

    // Display functions


private:
    // Static member functions
    static float Sigmoid_Function(float x);
    static float ReLU(float x);
    static float LeakyReLU(float x);
    static float ELU(float x, float alpha);
    static float Swish(float x);
    static float Tanh(float x);
    static float Linear_Activation(float x);
};

Matrix* Neural_Layer_Maker(int neurones_In_First_Layer, int neurones_In_Second_Layer,Matrix inputMatrix);
void  Forward_Pass(std::vector<Neural_Layer>&neural_layers, std::initializer_list<int> layers);
void Forward_Pass(std::pair<std::vector<Neural_Layer>, std::vector<Matrix>>& network_data, std::initializer_list<int> layers);
void Back_Propagation(std::vector<Neural_Layer>&neural_layers, std::initializer_list<int> layers,Matrix output,float &mean_squared_error);
void Back_Propagation(std::pair<std::vector<Neural_Layer>, std::vector<Matrix>>& network_data, std::initializer_list<int> layers, Matrix output, float &mean_squared_error) ;
void Matrix_Transpose_v2(Matrix *final, Matrix original) ;
void Matrix_Multiply_V2(Matrix *finalMatrix, Matrix firstMatrix, Matrix secondMatrix);
void Display_Gradients(const std::vector<Neural_Layer>&neural_layers, std::initializer_list<int> layers);
Matrix Matrix_Maker_2DArray_v2(int maxColumns, int totalRows, int desiredRows, int desiredColumns, int stride, int step, const float *data);
void Matrix_Broadcast(Matrix *result, Matrix original, int newRows,int newColumns);
void Matrix_Scalar_Multiply(Matrix &matrix, float scalar);
float Matrix_Sum_All_Elements(const Matrix& matrix);
void Matrix_Power(Matrix& matrix, float power);
void Matrix_Hadamard_Product(Matrix &result, const Matrix &a, const Matrix &b);
void Matrix_Absolute(Matrix &matrix);
void Learn(std::vector<Neural_Layer> &neural_layers, std::initializer_list<int> layers, Matrix output_matrix, float learning_rate, int iterations);
void fillMatrix(Matrix& matrix, float value);
std::vector<Neural_Layer> Form_Network(std::initializer_list<int> layers, Matrix inputMatrix);
std::pair<std::vector<Neural_Layer>, std::vector<Matrix>> Form_Network(std::initializer_list<int> layers, Matrix inputMatrix, const char* learning_algorithm);
void Learn(std::vector<Neural_Layer>&neural_layers, std::initializer_list<int> layers,Matrix input_matrix, Matrix output_matrix, float learning_rate, int iterations);
void Learn(std::pair<std::vector<Neural_Layer>, std::vector<Matrix>>& network_data, std::initializer_list<int> layers,  Matrix output_matrix, float learning_rate, float momentum, int iterations);
void Matrix_Copy(Matrix *destination, const Matrix *source);
void Matrix_Sum_Columns(Matrix &dest, const Matrix &src);
void Matrix_Fill(Matrix& matrix, float value);
void Matrix_Subtract_V2(Matrix& result, const Matrix& matrix1, const Matrix& matrix2);
std::vector<Neural_Layer> Form_Network(std::initializer_list<int> layers, int sample_size);
//Creates a matrix with a random number of elements, this matrix can also control the weight scaling
Matrix Matrix_Create_Random_V2(int rows, int columns, int scale);
Matrix Matrix_Create_Xavier_Uniform(int rows, int columns);
static void displayLayerDetails(const std::vector<Neural_Layer>& layers, int index);


#endif //DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_NEURAL_NETWORK_FRAMEWORK_H