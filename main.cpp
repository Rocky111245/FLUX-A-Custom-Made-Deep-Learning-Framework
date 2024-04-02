#include <iostream>
#include <initializer_list>
#include <vector>
#include <cmath>

extern "C" {
#include "library.h"
}

Matrix* Neural_Layer_Maker(int neurones_In_First_Layer, int neurones_In_Second_Layer,Matrix inputMatrix);
void forwardPass(std::initializer_list<int> layers,Matrix inputMatrix);

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
//    Neural_Layer_Maker(4,1,inputs);
//
//    std::cout<<"Hi, my name is Rakib"<<std::endl;
    forwardPass({4,3,2,1},inputs);
}

#include <cmath> // For std::exp

class Neural_Layer {
public:
    Matrix weights;
    Matrix bias;
    Matrix inputMatrix;
    Matrix weightedSum;
    Matrix activatedOutput;

    Neural_Layer() {
        // Default constructor body (can be empty)
    }
    Neural_Layer(int neuronsInFirstLayer, int neuronsInSecondLayer, const Matrix& inputMatrix)
            : inputMatrix(inputMatrix),
              weights(Matrix_Create_Random(neuronsInSecondLayer, neuronsInFirstLayer, 2)),
              bias(Matrix_Create_Random(neuronsInSecondLayer, 1, 2)),
              weightedSum(Matrix_Create_Zero(neuronsInSecondLayer, 1)),
              activatedOutput(Matrix_Create_Zero(neuronsInSecondLayer, 1)) {}

    void computeWeightedSum() {
        Matrix_Multiply(&weightedSum, weights, inputMatrix);
        Matrix_Add(&weightedSum, weightedSum, bias);
    }

    void activate() {
        for (int i = 0; i < weightedSum.row; ++i) {
            activatedOutput.data[i] = ReLU(weightedSum.data[i]);

        }
    }
    void activateLast(){
        for (int i = 0; i < weightedSum.row; ++i) {
            activatedOutput.data[i] = sigmoidFunction(weightedSum.data[i]);
        }
    }

    void displayInputs() const{
        Matrix_Display(inputMatrix);
    }
    void displayWeights() const{
        Matrix_Display(weights);

    }
    void displayWeightedSum() const {
        Matrix_Display(weightedSum);
    }

    void displayBias() const {
        Matrix_Display(bias);
    }

    void displayActivatedOutput() const {
        Matrix_Display(activatedOutput);
    }

private:
    static float sigmoidFunction(float x) {
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





//this function will iterate the Neural_Layer_Maker function on the number of networks I want to make

void forwardPass(std::initializer_list<int> layers, Matrix inputMatrix) {
    // Convert initializer_list to vector for easier access
    std::vector<int> layerSizes = layers;
    size_t size = layerSizes.size();
    std::cout<<"Size: "<<size<<std::endl;

    std::vector<Neural_Layer> neural_layers(size - 1);//layers will always be n-1 because the input layer is not considered
    // Initialize input matrix for the first layer
    Matrix currentInput = inputMatrix;

    // Iterate over each layer
    for (size_t i = 0; i < size - 1; ++i) {
        // Create a neural layer with appropriate size and input
        neural_layers[i]= Neural_Layer(layerSizes[i], layerSizes[i + 1], currentInput);
        printf("Size: %zu\n", neural_layers.size()); // Print the size of the vectorNeural_Node.size();
        printf("Capacity: %zu\n", neural_layers.capacity()); // Print the capacity of the vectorNeural_Node.capacity();
        std::cout<<"--------------"<<std::endl;
        // Compute weighted sum and activate the layer
        neural_layers[i].computeWeightedSum();
        neural_layers[i].activate();

        // Display outputs for debugging or analysis
        std::cout<<"Layer"<<i+1<<std::endl;
        neural_layers[i].displayInputs();
        neural_layers[i].displayWeights();
        neural_layers[i].displayWeightedSum();
        neural_layers[i].displayBias();

        if(i==size-2){
            neural_layers[i].activateLast();
        }
        else{
            neural_layers[i].activate();
        }
        neural_layers[i].displayActivatedOutput();
        std::cout<<"--------------"<<std::endl;


        // Set the output of this layer as the input for the next layer
        currentInput = neural_layers[i].activatedOutput;
    }
}


