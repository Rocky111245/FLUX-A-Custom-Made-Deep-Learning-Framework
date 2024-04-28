//
// Created by rakib on 27/4/2024.
//

#ifndef DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_EXCEL_H
#define DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_EXCEL_H
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
extern "C" {
#include "library.h"
}
float* Read_Csv_to_Array(const std::string& filename, int num_rows, int num_columns);
Matrix From_CSV_To_Matrix(const std::string& filename, int num_rows, int num_columns);
void Matrix_Create_From_CSV(Matrix &destination, const float *source) ;
void Normalize_Data(Matrix& matrix);
float* Read_Fixed_Width_to_Array(const std::string& filename, int num_rows, int num_columns, int column_width);
void From_CSV_Width_to_Matrix(const std::string& filename, int num_rows, int num_columns, int column_width);
#endif //DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_EXCEL_H
