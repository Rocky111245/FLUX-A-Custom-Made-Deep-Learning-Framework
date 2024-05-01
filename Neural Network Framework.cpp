#include "Neural Network Framework.h"
#include "Helper.h"
#include "Adaptive Learning Algorithms.h"


// Default constructor
Neural_Layer::Neural_Layer() {
    // Default constructor body (can be empty or initialized here)
}

// Parameterized constructor
Neural_Layer::Neural_Layer(int neuronsInFirstLayer, int neuronsInSecondLayer, const Matrix& inputMatrix)
        : neurons_in_first_layer(neuronsInFirstLayer), neurons_in_second_layer(neuronsInSecondLayer),
          input_matrix(inputMatrix),
          weights_matrix(Matrix_Create_Xavier_Uniform(neuronsInFirstLayer, neuronsInSecondLayer)),
          bias_matrix(Matrix_Create_Zero(1, neuronsInSecondLayer)), // Initialize biases to zero,
          weight_and_input_matrix(Matrix_Create_Zero(inputMatrix.row, weights_matrix.column)),
          weight_input_bias_matrix(Matrix_Create_Zero(inputMatrix.row, weights_matrix.column)),
          activated_output_matrix(Matrix_Create_Zero(inputMatrix.row, weights_matrix.column)),
          dC_da_matrix(Matrix_Create_Zero(inputMatrix.row, weights_matrix.column)),
          dh_da_matrix(Matrix_Create_Zero(inputMatrix.row, weights_matrix.column)),
          dC_dw_matrix(Matrix_Create_Zero(neuronsInFirstLayer, neuronsInSecondLayer)),
          dC_db_matrix(Matrix_Create_Zero(1, neuronsInSecondLayer))
{}

void Neural_Layer::Compute_Weighted_Sum() {
    Matrix_Multiply(&weight_and_input_matrix, input_matrix,weights_matrix);
    Matrix Broadcasted_Bias= Matrix_Create_Zero(weight_and_input_matrix.row,weight_and_input_matrix.column);
    Matrix_Broadcast(&Broadcasted_Bias,bias_matrix,weight_and_input_matrix.row,weight_and_input_matrix.column);
    Matrix_Add(&weight_input_bias_matrix, weight_and_input_matrix, Broadcasted_Bias);
    Matrix_Free(&Broadcasted_Bias);
}

void Neural_Layer::Activate() {
    // Iterate over each element of the matrix
    for (int i = 0; i < weight_input_bias_matrix.row; ++i) {
        for (int j = 0; j < weight_input_bias_matrix.column; ++j) {
            // Compute the correct index for a 2D matrix stored in a 1D array
            int index = i * weight_input_bias_matrix.column + j;
            activated_output_matrix.data[index] = LeakyReLU(weight_input_bias_matrix.data[index]);
        }
    }
}

void Neural_Layer::Activate_Last()  {
    // Iterate over each element of the matrix
    for (int i = 0; i < weight_input_bias_matrix.row; ++i) {
        for (int j = 0; j < weight_input_bias_matrix.column; ++j) {
            // Compute the correct index for a 2D matrix stored in a 1D array
            int index = i * weight_input_bias_matrix.column + j;
            activated_output_matrix.data[index] = Linear_Activation(weight_input_bias_matrix.data[index]);
        }
    }
}


void Neural_Layer::Dh_Da_Function() {
    // Iterate over all elements of the matrix (rows x columns)
    for (int i = 0; i < weight_input_bias_matrix.row; ++i) {
        for (int j = 0; j < weight_input_bias_matrix.column; ++j) {
            int index = i * weight_input_bias_matrix.column + j;  // Calculate the index for a 1D array representation of the matrix
            dh_da_matrix.data[index] = (weight_input_bias_matrix.data[index] > 0) ? 1.0f : 0.01f;
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


// Assuming definition of Neural_Layer and Matrix is available
// For example purposes, I assume Neural_Layer constructor and Matrix_Create_Zero function are defined somewhere

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
}


std::pair<std::vector<Neural_Layer>, std::vector<Matrix>> Form_Network(std::initializer_list<int> layers, Matrix inputMatrix, const char* learning_algorithm) {
    std::vector<int> layerSizes = layers;
    size_t size = layerSizes.size();
    std::cout << "Size of Neural Layer: " << size << std::endl;
    Matrix currentInput = inputMatrix;

    std::vector<Neural_Layer> neural_layers(size); // layers will always be n-1 because the input layer is not considered
    std::vector<Matrix> velocity;


    for (int i = 0; i < size; i++) {
        // Create a neural layer with the appropriate size and input
        neural_layers[i] = Neural_Layer(currentInput.column, layerSizes[i], currentInput);
        Matrix_Display(neural_layers[i].weights_matrix);
//        std::cout<<neural_layers[i].weights_matrix.row<<std::endl;
//        std::cout<<neural_layers[i].weights_matrix.column<<std::endl;

        // Reset currentInput for the next layer
        currentInput = Matrix_Create_Zero(inputMatrix.row, layerSizes[i]);

        // Initialize the velocity vector only if using "NAG"
        if (strcmp(learning_algorithm, "NAG") == 0) {
            velocity.reserve(size);// this vector array will have the same number of layers as the neural network.
            velocity[i] = Matrix_Create_Zero(neural_layers[i].weights_matrix.row, neural_layers[i].weights_matrix.column);
//            std::cout<<"Initializing Weights for NAG"<<std::endl;
//            Matrix_Display(velocity[i]);
//            std::cout<<velocity[i].row<<std::endl;
//            std::cout<<velocity[i].column<<std::endl;
        }

    }

    return {neural_layers,velocity};
}


//this function will iterate the Neural_Layer_Maker function on the number of networks I want to make
void Forward_Pass(std::vector<Neural_Layer> &neural_layers, std::initializer_list<int> layers) {
    // Convert initializer_list to vector for easier access
    std::vector<int> layerSizes = layers;
    size_t size = layerSizes.size();
    Matrix currentInput;


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

//this function will iterate the Neural_Layer_Maker function on the number of networks I want to make
void Forward_Pass(std::pair<std::vector<Neural_Layer>, std::vector<Matrix>>& network_data, std::initializer_list<int> layers) {
    // Convert initializer_list to vector for easier access
    std::vector<int> layerSizes = layers;
    size_t size = layerSizes.size();
    Matrix currentInput;

    auto &[neural_layers,velocity]=network_data;

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
    //have to put condition here later


    neural_layers[size - 1].Initialize_dC_dy_Matrix();
    neural_layers[size - 1].Initialize_Cost_Function_Matrix();
    Matrix_Subtract_V2(neural_layers[size - 1].C, neural_layers[size - 1].activated_output_matrix, output);
    Matrix_Power(neural_layers[size - 1].C, 2);
    mean_squared_error = Matrix_Sum_All_Elements(neural_layers[size - 1].C);
    mean_squared_error = mean_squared_error / 4.0f;
    Matrix_Subtract_V2(neural_layers[size - 1].dC_dy_matrix, neural_layers[size - 1].activated_output_matrix, output);
    Matrix_Scalar_Multiply(neural_layers[size - 1].dC_dy_matrix, (2.0f / 4.0f));

    Matrix_Hadamard_Product(neural_layers[size - 1].dC_da_matrix, neural_layers[size - 1].dC_dy_matrix, neural_layers[size - 1].dh_da_matrix);
    int last_layer = (int) size - 1;  //last layer-->2

    for (int layer_number = last_layer; layer_number >= 0; layer_number--) {


        Matrix n = Matrix_Create_Zero(neural_layers[layer_number].weights_matrix.column, neural_layers[layer_number].weights_matrix.row);
        Matrix_Transpose_v2(&n, neural_layers[layer_number].weights_matrix);


//      find dC_da for each layer
        if (layer_number != 0) {
            Matrix temp_delta = Matrix_Create_Zero(neural_layers[layer_number].dC_da_matrix.row, n.column);
            Matrix_Multiply(&temp_delta, neural_layers[layer_number].dC_da_matrix, n);

            Matrix_Hadamard_Product(neural_layers[layer_number - 1].dC_da_matrix, temp_delta, neural_layers[layer_number - 1].dh_da_matrix);
            Matrix_Free(&temp_delta);
        }
        Matrix_Free(&n);

//      find dC_dw for each layer
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

//suppose a fully connected dense neural network with NAG algorithm
void Back_Propagation(std::pair<std::vector<Neural_Layer>, std::vector<Matrix>>& network_data, std::initializer_list<int> layers, Matrix output, float &mean_squared_error, float &momentum) {

    auto &[neural_layers,velocity]=network_data;


    // Convert initializer_list to vector for easier access

    // Assuming 'layers' is the std::initializer_list<int>
    std::vector<int> neuronesInCurrentLayer(layers);
    size_t size = neuronesInCurrentLayer.size();
    //have to put condition here later


    neural_layers[size - 1].Initialize_dC_dy_Matrix();
    neural_layers[size - 1].Initialize_Cost_Function_Matrix();
    Matrix_Subtract_V2(neural_layers[size - 1].C, neural_layers[size - 1].activated_output_matrix, output);
    Matrix_Power(neural_layers[size - 1].C, 2);
    mean_squared_error = Matrix_Sum_All_Elements(neural_layers[size - 1].C);
    mean_squared_error = mean_squared_error / 4.0f;
    Matrix_Subtract_V2(neural_layers[size - 1].dC_dy_matrix, neural_layers[size - 1].activated_output_matrix, output);
    Matrix_Scalar_Multiply(neural_layers[size - 1].dC_dy_matrix, (2.0f / 4.0f));

    Matrix_Hadamard_Product(neural_layers[size - 1].dC_da_matrix, neural_layers[size - 1].dC_dy_matrix, neural_layers[size - 1].dh_da_matrix);
    int last_layer = (int) size - 1;  //last layer-->2

    for (int layer_number = last_layer; layer_number >= 0; layer_number--) {

        //a zero matrix will be created according to the layer number
        Matrix temp_modified_velocity= Matrix_Create_Zero(neural_layers[layer_number].weights_matrix.row,neural_layers[layer_number].weights_matrix.column);
        NAG_First_Call(neural_layers,velocity,temp_modified_velocity,layer_number,momentum);


    }


}





void Learn(std::vector<Neural_Layer> &neural_layers, std::initializer_list<int> layers, Matrix output_matrix, float learning_rate, int iterations) {
    std::vector<int> layerSizes = layers;
    size_t size = layerSizes.size();

    // Perform a forward pass and backpropagation to calculate initial MSE
    Forward_Pass(neural_layers, layers);
    float mean_squared_error;
    Back_Propagation(neural_layers, layers, output_matrix, mean_squared_error);
    float initial_mse = mean_squared_error;

   // std::cout << "Initial MSE: " << initial_mse << std::endl;


    for (int i = 0; i < iterations; i++) {
        Forward_Pass(neural_layers, layers);
        Back_Propagation(neural_layers, layers, output_matrix, mean_squared_error);


        // Logging after each iteration
//        std::cout << "Iteration " << i + 1 << " MSE: " << mean_squared_error << std::endl;

        for (size_t j = 0; j < size; j++) {
            // Update weights
            Matrix_Scalar_Multiply(neural_layers[j].dC_dw_matrix, learning_rate);
            Matrix_Subtract_V2(neural_layers[j].weights_matrix, neural_layers[j].weights_matrix, neural_layers[j].dC_dw_matrix);

            // Update biases
            Matrix_Scalar_Multiply(neural_layers[j].dC_db_matrix, learning_rate);
            Matrix_Subtract_V2(neural_layers[j].bias_matrix, neural_layers[j].bias_matrix, neural_layers[j].dC_db_matrix);

            // Reset gradients
            Matrix_Fill(neural_layers[j].dC_dw_matrix, 0);
            Matrix_Fill(neural_layers[j].dC_db_matrix, 0);
        }

        if ((i + 1) % 10 == 0 || i == iterations - 1) {
//            // Additional logging for specified intervals
            std::cout << "Checkpoint at iteration " << i + 1 << " MSE: " << mean_squared_error << std::endl;
        }
    }
    std::cout << "Final MSE after " << iterations << " iterations: " << mean_squared_error << std::endl;
    std::cout << "Improvement from initial MSE: " << initial_mse - mean_squared_error << std::endl;
    std::cout<< "Final Model Weights : "<<std::endl;
    Matrix_Display(neural_layers[size-1].weights_matrix);
    std::cout<< "Final Model Bias : "<<std::endl;
    Matrix_Display(neural_layers[size-1].bias_matrix);
    std::cout<< "Final Model  : "<<std::endl;
    Matrix_Display(neural_layers[size-1].weight_input_bias_matrix);
    std::cout<< "Final Predicted Values : "<<std::endl;
    Matrix_Display(neural_layers[size-1].activated_output_matrix);
}

void Learn(std::pair<std::vector<Neural_Layer>, std::vector<Matrix>>& network_data, std::initializer_list<int> layers,  Matrix output_matrix, float learning_rate, float momentum, int iterations) {
    std::vector<int> layerSizes = layers;
    size_t size = layerSizes.size();
    auto &[neural_layers,velocity]=network_data;

    // Perform a forward pass and backpropagation to calculate initial MSE
    Forward_Pass(network_data, layers);
    float mean_squared_error;
    Back_Propagation(network_data, layers, output_matrix, mean_squared_error,momentum);
    float initial_mse = mean_squared_error;

    // std::cout << "Initial MSE: " << initial_mse << std::endl;


    for (int i = 0; i < iterations; i++) {
        Forward_Pass(network_data, layers);
        Back_Propagation(network_data, layers, output_matrix, mean_squared_error,momentum);


        // Logging after each iteration
//        std::cout << "Iteration " << i + 1 << " MSE: " << mean_squared_error << std::endl;

        for (size_t j = 0; j < size; j++) {
            // Update weights
            Matrix_Scalar_Multiply(neural_layers[j].dC_dw_matrix, learning_rate);
            NAG_Second_Call(network_data,te)

            // Update biases
            Matrix_Scalar_Multiply(neural_layers[j].dC_db_matrix, learning_rate);
            Matrix_Subtract_V2(neural_layers[j].bias_matrix, neural_layers[j].bias_matrix, neural_layers[j].dC_db_matrix);

            // Reset gradients
            Matrix_Fill(neural_layers[j].dC_dw_matrix, 0);
            Matrix_Fill(neural_layers[j].dC_db_matrix, 0);
        }

        if ((i + 1) % 10 == 0 || i == iterations - 1) {
//            // Additional logging for specified intervals
            std::cout << "Checkpoint at iteration " << i + 1 << " MSE: " << mean_squared_error << std::endl;
        }
    }
    std::cout << "Final MSE after " << iterations << " iterations: " << mean_squared_error << std::endl;
    std::cout << "Improvement from initial MSE: " << initial_mse - mean_squared_error << std::endl;
    std::cout<< "Final Model Weights : "<<std::endl;
    Matrix_Display(neural_layers[size-1].weights_matrix);
    std::cout<< "Final Model Bias : "<<std::endl;
    Matrix_Display(neural_layers[size-1].bias_matrix);
    std::cout<< "Final Model  : "<<std::endl;
    Matrix_Display(neural_layers[size-1].weight_input_bias_matrix);
    std::cout<< "Final Predicted Values : "<<std::endl;
    Matrix_Display(neural_layers[size-1].activated_output_matrix);
}



