#include <iostream>
#include <vector>
#include "Neural Network Framework.h"

int main(){
    // Seed the random number generator
    srand((unsigned) time(NULL));
    float data[][3]={
            {1,0,0},
            {0,1,0},
            {1,1,1},
            {0,0,0}
    };
    size_t sizeOfData=sizeof(data)/sizeof(data[0]);

    Matrix inputs= Matrix_Maker_2DArray(3, 4, 2, 0, &data[0][0]);
    Matrix_Display(inputs);
//    Matrix modifiedInput= Matrix_Create_Zero(inputs.column, inputs.row);
//
//    size_t dataSize=sizeof(modifiedInput.data);
//    Matrix_Display(modifiedInput);
//    printf("%zu",dataSize);
//    printf("----------\n");
    Matrix output= Matrix_Maker_2DArray(3,4,1,2,&data[0][0]);
    Matrix_Display(output);

//    // Assuming 'inputs' is already defined and is a Matrix
//    std::vector<Neural_Layer> network = Form_Network({ 2,3,1}, inputs);
////    float mse;
//    Learn(network,{2,3,1}, inputs, output,0.001, 5);
    std::vector<Neural_Layer> network = Form_Network({1}, inputs);
    Learn(network,{1},inputs,output,0.0001,120000);

//    Forward_Pass(network,{3, 1}, inputs);
//    Back_Propagation(network, {3,1},output,mse);
//    Display_Gradients(network,{3, 1});
//    std::cout<<std::endl;
//    std::cout<<mse<<std::endl;
}





