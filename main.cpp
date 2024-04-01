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
    Matrix inputs= Matrix_Maker_2DArray(3, 4, 2, 0, &data[0][0]);
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

    Neural_Layer(int neuronsInFirstLayer, int neuronsInSecondLayer, const Matrix& inputMatrix)
            : inputMatrix(inputMatrix),
              weights(Matrix_Create_Random(neuronsInSecondLayer, neuronsInFirstLayer, 50)),
              bias(Matrix_Create_Random(neuronsInSecondLayer, 1, 10)),
              weightedSum(Matrix_Create_Zero(neuronsInSecondLayer, 1)),
              activatedOutput(Matrix_Create_Zero(neuronsInSecondLayer, 1)) {}

    void computeWeightedSum() {
        Matrix_Multiply(&weightedSum, weights, inputMatrix);
        Matrix_Add(&weightedSum, weightedSum, bias);
    }

    void activate() {
        for (int i = 0; i < weightedSum.row; ++i) {
            activatedOutput.data[i] = sigmoidFunction(weightedSum.data[i]);
        }
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
};




//Matrix* Neural_Layer_Maker(int neurones_In_First_Layer, int neurones_In_Second_Layer,Matrix inputMatrix, Matrix newMatrix){
//    Matrix weights=Matrix_Create_Random(neurones_In_Second_Layer,neurones_In_First_Layer,50);
//    Matrix_Display(weights);
//    Matrix_Display(inputMatrix);
//    newMatrix= Matrix_Create_Zero(neurones_In_Second_Layer,1);
//    Matrix_Multiply(&newMatrix,weights,inputMatrix);
//    Matrix_Display(newMatrix);
//    Matrix bias= Matrix_Create_Random(neurones_In_Second_Layer,1,10);
//    Matrix_Display(bias);
//    Matrix_Add(&newMatrix,newMatrix,bias);
//    Matrix_Display(newMatrix);
//    return &newMatrix;
//}

//this function will iterate the Neural_Layer_Maker function on the number of networks I want to make

void forwardPass(std::initializer_list<int> layers,Matrix inputMatrix) {
    std::vector<int> layerSizes=layers;
    size_t size= sizeof(layerSizes)/sizeof(layerSizes[0]);
    Neural_Layer Layer(layerSizes[0],layerSizes[1],inputMatrix);
    Layer.computeWeightedSum();
    Layer.displayWeightedSum();
    Layer.activate();
    Layer.displayActivatedOutput();
    Layer.displayBias();
}

