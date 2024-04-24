#include "Neural Network Framework.h"


// Default constructor
Neural_Layer::Neural_Layer() {
    // Default constructor body (can be empty or initialized here)
}

// Parameterized constructor
Neural_Layer::Neural_Layer(int neuronsInFirstLayer, int neuronsInSecondLayer, const Matrix& inputMatrix)
        : neurons_in_first_layer(neuronsInFirstLayer), neurons_in_second_layer(neuronsInSecondLayer),
          input_matrix(inputMatrix),
          weights_matrix(Matrix_Create_Random(neuronsInFirstLayer, neuronsInSecondLayer, 2)),
//          transposed_weights_matrix(Matrix_Create_Zero(weights_matrix.column, weights_matrix.row)),
          bias_matrix(Matrix_Create_Random(1, neuronsInSecondLayer, 1)),
          weight_and_input_matrix(Matrix_Create_Zero(inputMatrix.row, weights_matrix.column)),
          weight_input_bias_matrix(Matrix_Create_Zero(inputMatrix.row, weights_matrix.column)),
          activated_output_matrix(Matrix_Create_Zero(inputMatrix.row, weights_matrix.column)),
          dC_da_matrix(Matrix_Create_Zero(inputMatrix.row, weights_matrix.column)),
          dh_da_matrix(Matrix_Create_Zero(inputMatrix.row, weights_matrix.column)),
          dC_dw_matrix(Matrix_Create_Zero(neuronsInFirstLayer, neuronsInSecondLayer)),
          dC_db_matrix(Matrix_Create_Zero(1, neuronsInSecondLayer))
{}

void Neural_Layer::Compute_Weighted_Sum() {
//    transposed_weights_matrix= Matrix_Create_Zero(weights_matrix.column,weights_matrix.row);
//    Matrix_Transpose(&transposed_weights_matrix,weights_matrix);
    Matrix_Multiply(&weight_and_input_matrix, input_matrix,weights_matrix);
//    Matrix_Display(bias_matrix);
//    Matrix_Display(dC_db_matrix);
    Matrix Broadcasted_Bias= Matrix_Create_Zero(weight_and_input_matrix.row,weight_and_input_matrix.column);
    Matrix_Broadcast(&Broadcasted_Bias,bias_matrix,weight_and_input_matrix.row,weight_and_input_matrix.column);
    Matrix_Add(&weight_input_bias_matrix, weight_and_input_matrix, Broadcasted_Bias);
//    std::cout<<"Broadcasted Bias"<<std::endl;
//    Matrix_Display(Broadcasted_Bias);
    Matrix_Free(&Broadcasted_Bias);
}

void Neural_Layer::Activate() {
    // Iterate over each element of the matrix
    for (int i = 0; i < weight_input_bias_matrix.row; ++i) {
        for (int j = 0; j < weight_input_bias_matrix.column; ++j) {
            // Compute the correct index for a 2D matrix stored in a 1D array
            int index = i * weight_input_bias_matrix.column + j;
            activated_output_matrix.data[index] = ReLU(weight_input_bias_matrix.data[index]);
        }
    }
}

void Neural_Layer::Activate_Last() {
    // Iterate over each element of the matrix
    for (int i = 0; i < weight_input_bias_matrix.row; ++i) {
        for (int j = 0; j < weight_input_bias_matrix.column; ++j) {
            // Compute the correct index for a 2D matrix stored in a 1D array
            int index = i * weight_input_bias_matrix.column + j;
            activated_output_matrix.data[index] = Sigmoid_Function(weight_input_bias_matrix.data[index]);
        }
    }
}


void Neural_Layer::Dh_Da_Function() {
    // Iterate over all elements of the matrix (rows x columns)
    for (int i = 0; i < weight_input_bias_matrix.row; ++i) {
        for (int j = 0; j < weight_input_bias_matrix.column; ++j) {
            int index = i * weight_input_bias_matrix.column + j;  // Calculate the index for a 1D array representation of the matrix
            dh_da_matrix.data[index] = (weight_input_bias_matrix.data[index] > 0) ? 1.0f : 0.0f;
        }
    }
}


void Neural_Layer:: Initialize_dC_dy_Matrix(){
    dC_dy_matrix= Matrix_Create_Zero(activated_output_matrix.row,activated_output_matrix.column);
}
void Neural_Layer::Initialize_Cost_Function_Matrix(){
    C=Matrix_Create_Zero(activated_output_matrix.row,activated_output_matrix.column);
}

float Neural_Layer::Sigmoid_Function(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

float Neural_Layer::ReLU(float x) {
    return std::max(0.0f, x);
}

float Neural_Layer::LeakyReLU(float x, float alpha) {
    return (x > 0) ? x : alpha * x;
}

float Neural_Layer::ELU(float x, float alpha) {
    return (x > 0) ? x : alpha * (std::exp(x) - 1);
}

float Neural_Layer::Swish(float x) {
    return x * (1.0f / (1.0f + std::exp(-x)));
}

float Neural_Layer::Tanh(float x) {
    return std::tanh(x);
}
static void displayLayerDetails(const std::vector<Neural_Layer>& layers, int index)
{
    std::cout<<"Layer Number: "<<index<<std::endl;
    std::cout<<"Input Matrix"<<std::endl;
    Matrix_Display(layers[index].input_matrix);
    std::cout<<"Weight Matrix"<<std::endl;
    Matrix_Display(layers[index].weights_matrix);
    std::cout<<"Bias Matrix"<<std::endl;
    Matrix_Display(layers[index].bias_matrix);
    std::cout<<"Weighted Sum Matrix"<<std::endl;
    Matrix_Display(layers[index].weight_and_input_matrix);
    std::cout<<"Weighted Sum + Bias Matrix"<<std::endl;
    Matrix_Display(layers[index].weight_input_bias_matrix);
    std::cout<<"Activated Output Matrix"<<std::endl;
    Matrix_Display(layers[index].activated_output_matrix);
    std::cout<<"dh_da Matrix"<<std::endl;
    Matrix_Display(layers[index].dh_da_matrix);
    std::cout<<"dC_da Matrix"<<std::endl;
    Matrix_Display(layers[index].dC_da_matrix);
    std::cout<<"dC_dw Matrix"<<std::endl;
    Matrix_Display(layers[index].dC_dw_matrix);
    std::cout<<"dC_db Matrix"<<std::endl;
    Matrix_Display(layers[index].dC_db_matrix);
}


std::vector<Neural_Layer> Form_Network(std::initializer_list<int> layers, Matrix inputMatrix) {
    // Convert initializer_list to vector for easier access
    std::vector<int> layerSizes = layers;
    size_t size = layerSizes.size();
    std::cout<<"Size of Neural Layer: "<< size<<std::endl;
    Matrix currentInput=inputMatrix;

    std::vector<Neural_Layer> neural_layers(size);//layers will always be n-1 because the input layer is not considered
    for (int i=0;i<size;i++){
        // Create a neural layer with the appropriate size and input
        neural_layers[i]= Neural_Layer(currentInput.column, layerSizes[i], currentInput);
        currentInput= Matrix_Create_Zero(inputMatrix.row,layerSizes[i]);
    }
    return neural_layers;
};


//this function will iterate the Neural_Layer_Maker function on the number of networks I want to make
void Forward_Pass(std::vector<Neural_Layer> &neural_layers, std::initializer_list<int> layers, Matrix inputMatrix) {
    // Convert initializer_list to vector for easier access
    std::vector<int> layerSizes = layers;
    size_t size = layerSizes.size();


    Matrix currentInput = inputMatrix;

    // Iterate over each layer
    for (size_t i = 0; i < size; i++) {

        // Print the size of the vectorNeural_Node.size();
        // Print the capacity of the vectorNeural_Node.capacity();
        // Compute weighted sum and Activate the layer
        neural_layers[i].Compute_Weighted_Sum();
        neural_layers[i].Activate();
        neural_layers[i].Dh_Da_Function();

        if (i == size - 1) {
            neural_layers[i].Activate_Last();
        } else {
            neural_layers[i].Activate();
        }
        // Set the output of this layer as the input for the next layer
        if(i<size-1){
            currentInput = neural_layers[i].activated_output_matrix;
            neural_layers[i+1].input_matrix=currentInput;
//            std::cout<<"Input to next Layer: "<< i+1<<std::endl;
//            Matrix_Display(currentInput);
        }
//        displayLayerDetails(neural_layers,i);



    }

}


//suppose a fully connected dense neural network
void Back_Propagation(std::vector<Neural_Layer> &neural_layers, std::initializer_list<int> layers, Matrix output, float &mean_squared_error) {

    // Convert initializer_list to vector for easier access

    // Assuming 'layers' is the std::initializer_list<int>
    std::vector<int> neuronesInCurrentLayer(layers);
    size_t size = neuronesInCurrentLayer.size();

    neural_layers[size - 1].Initialize_dC_dy_Matrix();
    neural_layers[size - 1].Initialize_Cost_Function_Matrix();;
    Matrix_Subtract_V2(neural_layers[size - 1].C, neural_layers[size - 1].activated_output_matrix, output);
    Matrix_Power(neural_layers[size - 1].C, 2);
    mean_squared_error = Matrix_Sum_All_Elements(neural_layers[size - 1].C);
    mean_squared_error = mean_squared_error / 4.0f;
    Matrix_Subtract_V2(neural_layers[size - 1].dC_dy_matrix, neural_layers[size - 1].activated_output_matrix, output);;
    Matrix_Scalar_Multiply(neural_layers[size - 1].dC_dy_matrix, (2.0f / 4.0f));;

    Matrix_Hadamard_Product(neural_layers[size - 1].dC_da_matrix, neural_layers[size - 1].dC_dy_matrix, neural_layers[size - 1].dh_da_matrix);
    Matrix_Absolute(neural_layers[size - 1].dC_da_matrix);
    int last_layer = (int) size - 1;  //last layer-->2

    for (int layer_number = last_layer; layer_number >= 0; layer_number--) {


        Matrix n = Matrix_Create_Zero(neural_layers[layer_number].weights_matrix.column, neural_layers[layer_number].weights_matrix.row);
        Matrix_Transpose_v2(&n, neural_layers[layer_number].weights_matrix);

        if (layer_number != 0) {
            Matrix temp_delta = Matrix_Create_Zero(neural_layers[layer_number].dC_da_matrix.row, n.column);
            Matrix_Multiply(&temp_delta, neural_layers[layer_number].dC_da_matrix, n);

            Matrix_Hadamard_Product(neural_layers[layer_number - 1].dC_da_matrix, temp_delta, neural_layers[layer_number - 1].dh_da_matrix);
        }
        Matrix_Free(&n);


        Matrix m = Matrix_Create_Zero(neural_layers[layer_number].input_matrix.column, neural_layers[layer_number].input_matrix.row);
        Matrix_Transpose_v2(&m, neural_layers[layer_number].input_matrix);


        Matrix_Multiply(&neural_layers[layer_number].dC_dw_matrix, m, neural_layers[layer_number].dC_da_matrix);
        Matrix_Free(&m);

        // Averaging the gradients by the batch size (assuming batch size is 4)
        Matrix_Scalar_Multiply(neural_layers[layer_number].dC_dw_matrix, 1.0f / 4.0f);

        Matrix_Sum_Columns(neural_layers[layer_number].dC_db_matrix, neural_layers[layer_number].dC_da_matrix);

        // Averaging dC_db for the batch
        Matrix_Scalar_Multiply(neural_layers[layer_number].dC_db_matrix, 1.0f / 4.0f);


    }

}


void Learn(std::vector<Neural_Layer> &neural_layers, std::initializer_list<int> layers, Matrix input_matrix, Matrix output_matrix, float learning_rate, int iterations) {
    std::vector<int> layerSizes = layers;
    size_t size = layerSizes.size();

    // Perform a forward pass and backpropagation to calculate initial MSE
    Forward_Pass(neural_layers, layers, input_matrix);
    float mean_squared_error;
    Back_Propagation(neural_layers, layers, output_matrix, mean_squared_error);
    float initial_mse = mean_squared_error;

    std::cout << "Initial MSE: " << initial_mse << std::endl;


    for (int i = 0; i < iterations; i++) {
        Forward_Pass(neural_layers, layers, input_matrix);
        Back_Propagation(neural_layers, layers, output_matrix, mean_squared_error);
        // Logging after each iteration
        std::cout << "Iteration " << i + 1 << " MSE: " << mean_squared_error << std::endl;

        for (size_t j = 0; j < size; j++) {
//            std::cout<<"Display Weights"<<std::endl;
//            Matrix_Display(neural_layers[j].weights_matrix);
//            std::cout<<"Display Bias"<<std::endl;
//            Matrix_Display(neural_layers[j].bias_matrix);
//            std::cout<<"Display Weight Derivatives"<<std::endl;
//            Matrix_Display(neural_layers[j].dC_dw_matrix);
//            std::cout<<"Display Bias Derivatives"<<std::endl;
//            Matrix_Display(neural_layers[j].dC_db_matrix);

            // Update weights
            Matrix_Scalar_Multiply(neural_layers[j].dC_dw_matrix, learning_rate);
//            std::cout<<"Display Weight Derivatives after multiply"<<std::endl;
//            Matrix_Display(neural_layers[j].dC_dw_matrix);
            Matrix_Subtract_V2(neural_layers[j].weights_matrix, neural_layers[j].weights_matrix, neural_layers[j].dC_dw_matrix);

            // Update biases
            Matrix_Scalar_Multiply(neural_layers[j].dC_db_matrix, learning_rate);
            Matrix_Subtract_V2(neural_layers[j].bias_matrix, neural_layers[j].bias_matrix, neural_layers[j].dC_db_matrix);

            // Reset gradients
            Matrix_Fill(neural_layers[j].dC_dw_matrix, 0);
            Matrix_Fill(neural_layers[j].dC_db_matrix, 0);
//
//            // Logging layer update information
//            std::cout << "Iteration " << i + 1 << ", Layer " << j + 1 << ", Weight and Bias updated." << std::endl;
//            std::cout<<"Display Weights after change"<<std::endl;
//            Matrix_Display(neural_layers[j].weights_matrix);
//            std::cout<<"Display Bias after change"<<std::endl;
//            Matrix_Display(neural_layers[j].bias_matrix);
//            std::cout<<"Display Weight Derivatives after change"<<std::endl;
//            Matrix_Display(neural_layers[j].dC_dw_matrix);
//            std::cout<<"Display Bias Derivatives after change"<<std::endl;
//            Matrix_Display(neural_layers[j].dC_db_matrix);
        }

        if ((i + 1) % 10 == 0 || i == iterations - 1) {
//            // Additional logging for specified intervals
//            std::cout << "Checkpoint at iteration " << i + 1 << " MSE: " << mean_squared_error << std::endl;
        }
    }
    std::cout << "Final MSE after " << iterations << " iterations: " << mean_squared_error << std::endl;
    std::cout << "Improvement from initial MSE: " << initial_mse - mean_squared_error << std::endl;
    std::cout<< "Final Model : "<<std::endl;
    Matrix_Display(neural_layers[size-1].activated_output_matrix);
}


void Display_Gradients(const std::vector<Neural_Layer> &neural_layers, std::initializer_list<int> layers) {
    // Convert initializer_list to vector for easier access
    // Assuming 'layers' is your std::initializer_list<int>
    std::vector<int> layerSizes = layers;
    size_t size = layerSizes.size();


    for (int layer_number = 0; layer_number < size; layer_number++) {

    }

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
    if (result->data == NULL) {
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
    if (M.data == NULL) {
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

// Function to copy data from source matrix to destination matrix.
void Matrix_Copy(Matrix *destination, const Matrix *source) {
    if (destination->row != source->row || destination->column != source->column) {

        return;
    }

    for (int i = 0; i < source->row; i++) {
        for (int j = 0; j < source->column; j++) {
            int index = Matrix_Index_Finder(source->column, i, j);
            destination->data[index] = source->data[index];
        }
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