//
// Created by rakib on 29/4/2024.
//
#ifndef DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_ADAPTIVE_LEARNING_ALGORITHMS_H
#define DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_ADAPTIVE_LEARNING_ALGORITHMS_H

#include <iostream>
#include <vector>
#include "Neural Network Framework.h"
#include "Excel.h"
#include "Helper.h"
extern "C" {
#include "library.h"
}



void NAG_First_Call(std::tuple<std::vector<Neural_Layer>, std::vector<Matrix>, std::vector<Matrix>>& network_data,int layer_number,float momentum);
void NAG_Second_Call(std::tuple<std::vector<Neural_Layer>, std::vector<Matrix>, std::vector<Matrix>>& network_data, size_t layer_number);






















#endif //DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_ADAPTIVE_LEARNING_ALGORITHMS_H
