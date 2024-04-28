
#ifndef DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_HELPER_H
#define DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_HELPER_H
#include <iostream>
#include <initializer_list>
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>
extern "C" {
#include "library.h"
}

Matrix Matrix_Create_Xavier_Uniform(int rows, int columns);
Matrix Matrix_Create_Random_V2(int rows, int columns, int scale);
void Matrix_Subtract_V2(Matrix& result, const Matrix& matrix1, const Matrix& matrix2);
void Matrix_Fill(Matrix &matrix, float value);
void Matrix_Sum_Columns(Matrix &dest, const Matrix &src);
void fillMatrix(Matrix &matrix, float value);
void Matrix_Absolute(Matrix &matrix);
void Matrix_Hadamard_Product(Matrix &result, const Matrix &a, const Matrix &b);
void Matrix_Power(Matrix &matrix, float power);
float Matrix_Sum_All_Elements(const Matrix &matrix);
void Matrix_Scalar_Multiply(Matrix &matrix, float scalar);
Matrix Matrix_Maker_2DArray_v2(int maxColumns, int totalRows, int desiredRows, int desiredColumns, int stride, int step, const float *data);
void Matrix_Transpose_v2(Matrix *final, Matrix original);
void Matrix_Broadcast(Matrix *result, Matrix original, int newRows, int newColumns);












#endif //DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_HELPER_H
