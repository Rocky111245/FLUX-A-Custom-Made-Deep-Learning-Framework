#include "Neural Network Framework.h"



// Default constructor
Neural_Layer::Neural_Layer() {
    // Default constructor body (can be empty or initialized here)
}

// Parameterized constructor
Neural_Layer::Neural_Layer(int neuronsInFirstLayer, int neuronsInSecondLayer, const Matrix& inputMatrix)
        : neurons_in_first_layer(neuronsInFirstLayer), neurons_in_second_layer(neuronsInSecondLayer),
          input_matrix(inputMatrix),
          weights_matrix(Initialize_Weights(neuronsInFirstLayer,neuronsInSecondLayer)),
          bias_matrix(Matrix(1, neuronsInSecondLayer)), // Initialize biases to zero,
          weight_and_input_matrix(Matrix(inputMatrix.rows(), weights_matrix.columns())),
          weight_input_bias_matrix(Matrix(inputMatrix.rows(), weights_matrix.columns())),
          activated_output_matrix(Matrix(inputMatrix.rows(), weights_matrix.columns())),
          dC_da_matrix(Matrix(inputMatrix.rows(), weights_matrix.columns())),
          dh_da_matrix(Matrix(inputMatrix.rows(), weights_matrix.columns())),
          dC_dw_matrix(Matrix(neuronsInFirstLayer, neuronsInSecondLayer)),
          dC_db_matrix(Matrix(1, neuronsInSecondLayer))
{}

void Neural_Layer::Compute_Weighted_Sum() {
    Matrix_Multiply(weight_and_input_matrix, input_matrix,weights_matrix);
    Matrix Broadcasted_Bias= Matrix(weight_and_input_matrix.rows(),weight_and_input_matrix.columns());
    Matrix_Broadcast(Broadcasted_Bias,bias_matrix,weight_and_input_matrix.rows(),weight_and_input_matrix.columns());
    Matrix_Add(weight_input_bias_matrix, weight_and_input_matrix, Broadcasted_Bias);
}

void Neural_Layer::Activate() {
    // Iterate over each element of the matrix
    for (int i = 0; i < weight_input_bias_matrix.rows(); ++i) {
        for (int j = 0; j < weight_input_bias_matrix.columns(); ++j) {
            activated_output_matrix(i,j) = LeakyReLU(weight_input_bias_matrix(i,j));
        }
    }
}

void Neural_Layer::Activate_Last()  {
    // Iterate over each element of the matrix
    for (int i = 0; i < weight_input_bias_matrix.rows(); ++i) {
        for (int j = 0; j < weight_input_bias_matrix.columns(); ++j) {
            activated_output_matrix(i,j) = Linear_Activation(weight_input_bias_matrix(i,j));
        }
    }
}


void Neural_Layer::Dh_Da_Function() {
    // Iterate over all elements of the matrix (rows x columns)
    for (int i = 0; i < weight_input_bias_matrix.rows(); ++i) {
        for (int j = 0; j < weight_input_bias_matrix.columns(); ++j) {
            dh_da_matrix(i,j) = (weight_input_bias_matrix(i,j)> 0) ? 1.0f : 0.01f;
        }
    }
}


//void Neural_Layer::Dh_Da_Function() {
//    // Iterate over all elements of the matrix (rows x columns)
//    for (int i = 0; i < weight_input_bias_matrix.row; ++i) {
//        for (int j = 0; j < weight_input_bias_matrix.column; ++j) {
//            int index = i * weight_input_bias_matrix.column + j;  // Calculate the index for a 1D array representation of the matrix
//            dh_da_matrix.data[index] = (weight_input_bias_matrix.data[index] > 0) ? 1.0f : 0.0f;
//        }
//    }
//}


void Neural_Layer::Initialize_dC_dy_Matrix() {
    dC_dy_matrix = std::make_unique<Matrix>(activated_output_matrix.rows(), activated_output_matrix.columns());
}
void Neural_Layer::Initialize_Cost_Function_Matrix(){
    C=std::make_unique<Matrix>(activated_output_matrix.rows(), activated_output_matrix.columns());
}

float Neural_Layer::Sigmoid_Function(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

float Neural_Layer::ReLU(float x) {
    return std::max(0.0f, x);
}

float Neural_Layer::LeakyReLU(float x) {
    float alpha = 0.01;  // Example value, can be changed to the desired slope for negative inputs
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

float Neural_Layer::Linear_Activation(float x) {
    return x;
}

Matrix Neural_Layer:: Initialize_Weights(int row, int column){
    Matrix A (row,column);
    Matrix_Randomize(A);
    return A;
}




//static void displayLayerDetails(const Neural_Layer_Information &layers , int index)
//{
//    const auto &layer=layers.neural_layers;
//
//    std::cout<<"Layer Number: "<<index<<std::endl;
//    std::cout<<"Input Matrix"<<std::endl;
//    Matrix::Print(layer);
//    std::cout<<"Weight Matrix"<<std::endl;
//    Matrix::Print(layers[index].weights_matrix);
//    std::cout<<"Bias Matrix"<<std::endl;
//    Matrix::Print(layers[index].bias_matrix);
//    std::cout<<"Weighted Sum Matrix"<<std::endl;
//    Matrix::Print(layers[index].weight_and_input_matrix);
//    std::cout<<"Weighted Sum + Bias Matrix"<<std::endl;
//    Matrix::Print(layers[index].weight_input_bias_matrix);
//    std::cout<<"Activated Output Matrix"<<std::endl;
//    Matrix::Print(layers[index].activated_output_matrix);
//    std::cout<<"dh_da Matrix"<<std::endl;
//    Matrix::Print(layers[index].dh_da_matrix);
//    std::cout<<"dC_da Matrix"<<std::endl;
//    Matrix::Print(layers[index].dC_da_matrix);
//    std::cout<<"dC_dw Matrix"<<std::endl;
//    Matrix::Print(layers[index].dC_dw_matrix);
//    std::cout<<"dC_db Matrix"<<std::endl;
//    Matrix::Print(layers[index].dC_db_matrix);
//}


// Assuming definition of Neural_Layer and Matrix is available
// For example purposes, I assume Neural_Layer constructor and Matrix_Create_Zero function are defined somewhere

Neural_Layer_Information Form_Network(std::initializer_list<int> layers, Matrix inputMatrix,const Matrix& outputMatrix) {
    std::vector<int> layerSizes = layers;
    size_t size = layerSizes.size();
    std::cout << "Size of Neural Layer: " << size << std::endl;
    int sampleSize = inputMatrix.rows();
    std::cout << "Size of Sample: " << sampleSize << std::endl;

    Matrix currentInput = std::move(inputMatrix);
    Matrix::Print(currentInput);

    std::vector<Neural_Layer> neural_layers;
    neural_layers.reserve(size);

    for (size_t i = 0; i < size; i++) {
        neural_layers.emplace_back(currentInput.columns(), layerSizes[i], currentInput);
        currentInput = Matrix(currentInput.rows(), layerSizes[i]);
    }

    return Neural_Layer_Information{std::move(neural_layers), layerSizes,sampleSize,outputMatrix};
}



//this function will iterate the Neural_Layer_Maker function on the number of networks I want to make
void Forward_Pass(Neural_Layer_Information &neural_layer_information) {
    //Separated for readability
    auto &neural_layers=neural_layer_information.neural_layers;
    auto &layers=neural_layer_information.layers_vector;

    size_t size = layers.size();
    Matrix currentInput;

    // Iterate over each layer
    for (size_t i = 0; i < size; i++) {
        // Compute weighted sum and activate the layer
        neural_layers[i].Compute_Weighted_Sum();
        neural_layers[i].Activate();
        neural_layers[i].Dh_Da_Function();

        // Set the output of this layer as the input for the next layer
        if (i < size - 1) {
            Matrix_Resize(currentInput, neural_layers[i].activated_output_matrix.rows(), neural_layers[i].activated_output_matrix.columns());
            currentInput = neural_layers[i].activated_output_matrix;
            neural_layers[i + 1].input_matrix = currentInput;
        } else if (i == size - 1) {
            neural_layers[i].Activate_Last();
        }

    }

}



//suppose a fully connected dense neural network
void Back_Propagation(Neural_Layer_Information &neural_layer_information, float &mean_squared_error) {

    //Separated for readability
    auto &neural_layers=neural_layer_information.neural_layers;
    const auto &layers=neural_layer_information.layers_vector;
    const auto &sample_size=neural_layer_information.sample_size;
    const auto & output_matrix=neural_layer_information.outputMatrix;


    size_t size = layers.size();
    //have to put condition here later


    //Initializing the extra matrices for the last layer before moving onto the actual backpropagation part
    neural_layers[size - 1].Initialize_dC_dy_Matrix();
    neural_layers[size - 1].Initialize_Cost_Function_Matrix();

    auto &dC_dy=*neural_layers[size-1].dC_dy_matrix;
    auto &C=*neural_layers[size-1].C;
    auto &dh_da=neural_layers[size-1].dh_da_matrix;
    auto &dC_da=neural_layers[size-1].dC_da_matrix;





    Matrix_Subtract(C, neural_layers[size - 1].activated_output_matrix, output_matrix);
    Matrix_Power(C, 2);

    mean_squared_error = Matrix_Sum_All_Elements(C);
    mean_squared_error = mean_squared_error / sample_size;


    Matrix_Subtract(dC_dy, neural_layers[size - 1].activated_output_matrix, output_matrix);
    Matrix_Scalar_Multiply(dC_dy, (2.0f / sample_size));
    Matrix_Hadamard_Product(dC_da,dC_dy, dh_da);
    //dC_da of the last layer is prepared and this will help in the calculation as we go backwards


    //Work Backwards from Last Layer
    int last_layer = (int) size - 1;  //last layer-->2

    for (int layer_number = last_layer; layer_number >= 0; layer_number--) {


        Matrix n = Matrix(neural_layers[layer_number].weights_matrix.columns(), neural_layers[layer_number].weights_matrix.rows());
        Matrix_Transpose(n, neural_layers[layer_number].weights_matrix);


//      find dC_da for each layer
        if (layer_number != 0) {
            Matrix temp_delta = Matrix(neural_layers[layer_number].dC_da_matrix.rows(), n.columns());
            Matrix_Multiply(temp_delta, neural_layers[layer_number].dC_da_matrix, n);

            Matrix_Hadamard_Product(neural_layers[layer_number - 1].dC_da_matrix, temp_delta, neural_layers[layer_number - 1].dh_da_matrix);
        }

//      find dC_dw for each layer
        Matrix m = Matrix(neural_layers[layer_number].input_matrix.columns(), neural_layers[layer_number].input_matrix.rows());
        Matrix_Transpose(m, neural_layers[layer_number].input_matrix);

        Matrix_Multiply(neural_layers[layer_number].dC_dw_matrix, m, neural_layers[layer_number].dC_da_matrix);


        // Averaging the gradients by the batch size (assuming batch size is 4)
        Matrix_Scalar_Multiply(neural_layers[layer_number].dC_dw_matrix, 1.0f / sample_size);

        Matrix_Sum_Columns(neural_layers[layer_number].dC_db_matrix, neural_layers[layer_number].dC_da_matrix);

        // Averaging dC_db for the batch
        Matrix_Scalar_Multiply(neural_layers[layer_number].dC_db_matrix, 1.0f / sample_size);

    }

}



void Learn(Neural_Layer_Information &neural_layer_information, float learning_rate, int iterations) {

    //Separated for readability
    auto &neural_layers=neural_layer_information.neural_layers;
    auto &layers=neural_layer_information.layers_vector;
    auto &sample_size=neural_layer_information.sample_size;
    auto &output_matrix=neural_layer_information.outputMatrix;


    size_t size = layers.size();

    // Perform a forward pass and backpropagation to calculate initial MSE
    Forward_Pass(neural_layer_information);
    float mean_squared_error;
    Back_Propagation(neural_layer_information, mean_squared_error);
    float initial_mse = mean_squared_error;

    // std::cout << "Initial MSE: " << initial_mse << std::endl;


    for (int i = 0; i < iterations; i++) {
        Forward_Pass(neural_layer_information);
        Back_Propagation(neural_layer_information, mean_squared_error);;


        // Logging after each iteration
//        std::cout << "Iteration " << i + 1 << " MSE: " << mean_squared_error << std::endl;

        for (size_t j = 0; j < size; j++) {
            // Update weights
            Matrix_Scalar_Multiply(neural_layers[j].dC_dw_matrix, learning_rate);
            Matrix temp_weight(neural_layers[j].weights_matrix.rows(),neural_layers[j].weights_matrix.columns());
            Matrix_Subtract(temp_weight, neural_layers[j].weights_matrix, neural_layers[j].dC_dw_matrix);
            neural_layers[j].weights_matrix=std::move(temp_weight);

            // Update biases
            Matrix_Scalar_Multiply(neural_layers[j].dC_db_matrix, learning_rate);
            Matrix temp_bias(neural_layers[j].bias_matrix.rows(),neural_layers[j].bias_matrix.columns());
            Matrix_Subtract(temp_bias, neural_layers[j].bias_matrix, neural_layers[j].dC_db_matrix);
            neural_layers[j].bias_matrix=std::move(temp_bias);

            // Reset gradients
            Matrix_Fill(neural_layers[j].dC_dw_matrix,0);
            Matrix_Fill(neural_layers[j].dC_db_matrix,0);
        }

        if ((i + 1) % 10 == 0 || i == iterations - 1) {
//            // Additional logging for specified intervals
            std::cout << "Checkpoint at iteration " << i + 1 << " MSE: " << mean_squared_error << std::endl;
        }
    }
    std::cout << "Final MSE after " << iterations << " iterations: " << mean_squared_error << std::endl;
    std::cout << "Improvement from initial MSE: " << initial_mse - mean_squared_error << std::endl;
    std::cout<< "Final Model Weights : "<<std::endl;
    Matrix::Print(neural_layers[size-1].weights_matrix);
    std::cout<< "Final Model Bias : "<<std::endl;
    Matrix::Print(neural_layers[size-1].bias_matrix);
    std::cout<< "Final Model  : "<<std::endl;
    Matrix::Print(neural_layers[size-1].weight_input_bias_matrix);
    std::cout<< "Final Predicted Values : "<<std::endl;
    Matrix::Print(neural_layers[size-1].activated_output_matrix);
}



void Matrix_Fill(Matrix& matrix, float value) {
    for (int i = 0; i < matrix.rows(); ++i) {
        for (int j = 0; j < matrix.columns(); ++j) {
            matrix(i, j) = value;
        }
    }
}
