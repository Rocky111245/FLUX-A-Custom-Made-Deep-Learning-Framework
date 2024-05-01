#include "Helper.h"
//Temporary page containing Functions which will later be shifted to the Custom CMatrix Library.

Matrix Matrix_Create_Xavier_Uniform(int rows, int columns) {
    Matrix M;
    M.row = rows;
    M.column = columns;
    M.data = new float[rows * columns];
    if (M.data == nullptr) {
        std::cerr << "Failed to allocate memory for matrix data.\n";
        exit(EXIT_FAILURE);
    }

    // Calculate the scaling factor according to Xavier initialization for uniform distribution
    float limit = std::sqrt(6.0f / (rows + columns));  // n_in = rows, n_out = columns for forward pass
    // Use C++11 random number generation
    std::mt19937 gen(std::random_device{}()); // Standard mersenne_twister_engine seeded with random_device
    std::uniform_real_distribution<float> dis(-limit, limit);


    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            int index = i * columns + j;
            // Generate a random float between -limit and limit using C++11 <random>
            M.data[index] = dis(gen);
        }
    }
    return M;
}




// Broadcasts an existing matrix to a larger size specified by newRows and newColumns
void Matrix_Broadcast(Matrix *result, const Matrix original, const int newRows, const int newColumns) {
    if (newRows % original.row != 0 || newColumns % original.column != 0) {
        exit(1);
    }

    // Allocate result matrix with new dimensions
    result->row = newRows;
    result->column = newColumns;
    result->data = (float *) malloc(newRows * newColumns * sizeof(float));
    if (result->data == nullptr) {
        exit(1);
    }

    // Fill the new matrix by repeating the original matrix values
    for (int i = 0; i < newRows; i++) {
        for (int j = 0; j < newColumns; j++) {
            int originalRow = i % original.row;
            int originalColumn = j % original.column;
            int originalIndex = Matrix_Index_Finder(original.column, originalRow, originalColumn);
            int newIndex = Matrix_Index_Finder(newColumns, i, j);
            result->data[newIndex] = original.data[originalIndex];
        }
    }
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



// Modified matrix maker with stride for columns and corrected step for rows
Matrix Matrix_Maker_2DArray_v2(int maxColumns, int totalRows, int desiredRows, int desiredColumns, int stride, int step, const float *data) {
    Matrix M;
    M.row = desiredRows;
    M.column = desiredColumns;
    M.data = (float *) calloc(desiredRows * desiredColumns, sizeof(float));
    if (M.data == nullptr) {
        std::cerr << "Failed to allocate memory for matrix data.\n";
        exit(EXIT_FAILURE);
    }

    int index = 0;
    // Iterate through rows with a step
    for (int i = 0; i < desiredRows && (i + step) < totalRows; i++) {
        int baseRow = i + step; // Adjust baseRow calculation to account for step directly
        for (int j = stride; j < desiredColumns + stride; j++) {
            if (j < maxColumns) { // Ensure that we do not exceed column bounds
                int dataIndex = Matrix_Index_Finder(maxColumns, baseRow, j);
                M.data[index++] = data[dataIndex];
            } else {
                // Handle the case where column index exceeds the bounds
                M.data[index++] = 0; // Optionally fill with a default value or handle as needed
            }
        }
    }
    return M;
}

// Function to multiply all elements of a matrix by a scalar value
void Matrix_Scalar_Multiply(Matrix &matrix, float scalar) {
    for (int i = 0; i < matrix.row; ++i) {
        for (int j = 0; j < matrix.column; ++j) {
            int index = i * matrix.column + j;  // Calculate the index for a 1D array representation of the matrix
            matrix.data[index] *= scalar;       // Multiply the element by the scalar value
        }
    }
}

// Function to add up all values in a matrix and return a single scalar
float Matrix_Sum_All_Elements(const Matrix &matrix) {
    float totalSum = 0;
    for (int i = 0; i < matrix.row; i++) {
        for (int j = 0; j < matrix.column; j++) {
            totalSum += matrix.data[i * matrix.column + j];
        }
    }
    return totalSum;
}

// Function to raise each element of a matrix to a specified power
void Matrix_Power(Matrix &matrix, float power) {
    for (int i = 0; i < matrix.row; i++) {
        for (int j = 0; j < matrix.column; j++) {
            int index = i * matrix.column + j;
            matrix.data[index] = std::pow(matrix.data[index], power);
        }
    }
}


// Function to perform Hadamard Product (element-wise multiplication)
void Matrix_Hadamard_Product(Matrix &result, const Matrix &a, const Matrix &b) {
    // First, check if the input matrices have the same dimensions
    if (a.row != b.row || a.column != b.column) {
        return;
    }

    // Ensure the result matrix has the correct dimensions
    if (result.row != a.row || result.column != a.column) {
        return;
    }

    // Perform element-wise multiplication
    for (int i = 0; i < a.row; i++) {
        for (int j = 0; j < a.column; j++) {
            int index = Matrix_Index_Finder(a.column, i, j);  // Correctly calculate index based on matrix dimensions
            result.data[index] = a.data[index] * b.data[index];
        }
    }
}


void Matrix_Absolute(Matrix &matrix) {
    for (int i = 0; i < matrix.row; ++i) {
        for (int j = 0; j < matrix.column; ++j) {
            int index = i * matrix.column + j;  // Calculate the index for a 1D array representation of the matrix
            matrix.data[index] = std::abs(matrix.data[index]);  // Apply the abs function to each element
        }
    }
}

void fillMatrix(Matrix &matrix, float value) {
    int totalElements = matrix.row * matrix.column;
    for (int i = 0; i < totalElements; i++) {
        matrix.data[i] = value;
    }
}



void Matrix_Sum_Columns(Matrix &dest, const Matrix &src) {
    // Ensure the destination matrix has the correct number of columns
    if (dest.column != src.column) {

        return;
    }

    // Calculate the sum of each column and assign it to every row of that column in the destination matrix
    for (int col = 0; col < src.column; ++col) {
        float column_sum = 0;
        for (int row = 0; row < src.row; ++row) {
            column_sum += src.data[row * src.column + col];
        }
        for (int row = 0; row < dest.row; ++row) {
            dest.data[row * dest.column + col] = column_sum;
        }
    }
}

// Helper function to fill a matrix with a specific value
void Matrix_Fill(Matrix &matrix, float value) {
    for (int i = 0; i < matrix.row * matrix.column; i++) {
        matrix.data[i] = value;
    }
}

// Function to subtract one matrix from another
void Matrix_Subtract_V2(Matrix& result, const Matrix& matrix1, const Matrix& matrix2) {
    // Ensure both matrices have the same dimensions
    if (matrix1.row != matrix2.row || matrix1.column != matrix2.column) {
        throw std::invalid_argument("Error: Matrices dimensions do not match.");
    }

    // Assuming 'result' matrix is pre-allocated and has the same dimensions as 'matrix1' and 'matrix2'
    for (int i = 0; i < matrix1.row; i++) {
        for (int j = 0; j < matrix1.column; j++) {
            int index = Matrix_Index_Finder(matrix1.column, i, j);
            result.data[index] = matrix1.data[index] - matrix2.data[index];
        }
    }
}

//Creates a matrix with a random number of elements, this matrix can also control the weight scaling
Matrix Matrix_Create_Random_V2(int rows, int columns, int scale) {
    Matrix M;
    M.row = rows;
    M.column = columns;
    M.data = (float*) malloc(rows * columns * sizeof(float));
    if (M.data == nullptr) {
        exit(1); // Exit if memory allocation fails
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            int index = Matrix_Index_Finder(columns, i, j);
            // Generate a random float between -scale/2 and scale/2
            M.data[index] = (rand() / (float)RAND_MAX) * scale - (scale / 2.0f);
        }
    }
    return M;
}


void Matrix_DeepCopy(Matrix& destination,const Matrix& source) {
    // Check if dimensions match
    if (destination.row != source.row || destination.column != source.column) {
        delete[] destination.data;  // Free existing destination data
        destination.data = new float[source.row * source.column];  // Allocate new memory
        destination.row = source.row;
        destination.column = source.column;
    }

    // Perform the copy
    std::copy(source.data, source.data + source.row * source.column, destination.data);
}

