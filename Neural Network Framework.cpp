#include "Neural Network Framework.h"
#include <string>


// Default constructor
Neural_Layer::Neural_Layer() {
    // Default constructor body (can be empty or initialized here)
}

// Parameterized constructor
// Constructor
Neural_Layer::Neural_Layer(int neuronsInFirstLayer, int neuronsInSecondLayer, const Matrix& inputMatrix, ActivationType hidden_activation, ActivationType output_activation)
        : neurons_in_first_layer(neuronsInFirstLayer),
          neurons_in_second_layer(neuronsInSecondLayer),
          input_matrix(inputMatrix),
          weights_matrix(Initialize_Weights(neuronsInFirstLayer, neuronsInSecondLayer)),
          bias_matrix(Matrix(1, neuronsInSecondLayer)), // Initialize biases to zero
          weight_and_input_matrix(Matrix(inputMatrix.rows(), weights_matrix.columns())),
          weight_input_bias_matrix(Matrix(inputMatrix.rows(), weights_matrix.columns())),
          activated_output_matrix(Matrix(inputMatrix.rows(), weights_matrix.columns())),
          dC_da_matrix(Matrix(inputMatrix.rows(), weights_matrix.columns())),
          dh_da_matrix(Matrix(inputMatrix.rows(), weights_matrix.columns())),
          dC_dw_matrix(Matrix(neuronsInFirstLayer, neuronsInSecondLayer)),
          dC_db_matrix(Matrix(1, neuronsInSecondLayer)),
          layer_activation{hidden_activation, output_activation}
{}

//this function does both the summation and activation part inside a neural node
void Neural_Layer::Compute_Weighted_Sum() {
    Matrix_Multiply(weight_and_input_matrix, input_matrix,weights_matrix);
    Matrix Broadcasted_Bias= Matrix(weight_and_input_matrix.rows(),weight_and_input_matrix.columns());
    Matrix_Broadcast(Broadcasted_Bias,bias_matrix,weight_and_input_matrix.rows(),weight_and_input_matrix.columns());
    Matrix_Add(weight_input_bias_matrix, weight_and_input_matrix, Broadcasted_Bias);
}

// Function to convert ActivationType enum to string
std::string ActivationTypeToString(ActivationType activation_type) {
    switch (activation_type) {
        case ActivationType::RELU:
            return "RELU";
        case ActivationType::SIGMOID:
            return "SIGMOID";
        case ActivationType::TANH:
            return "TANH";
        case ActivationType::LEAKY_RELU:
            return "LEAKY_RELU";
        case ActivationType::SWISH:
            return "SWISH";
        case ActivationType::LINEAR:
            return "LINEAR";
        default:
            return "UNKNOWN";
    }
}

//this function is responsible for activating the WIB (weights,input,bias) matrix
void Neural_Layer::Activate(ActivationType activation_type) {
    // Iterate over each element of the weight_input_bias_matrix (rows x columns)
    for (int i = 0; i < weight_input_bias_matrix.rows(); ++i) {
        for (int j = 0; j < weight_input_bias_matrix.columns(); ++j) {
            float value = weight_input_bias_matrix(i, j);  // Fetch the value only once per loop

            // Apply the appropriate activation function based on the activation type
            switch (activation_type) {
                case ActivationType::RELU:
                    // ReLU activation
                    activated_output_matrix(i, j) = ReLU(value);
                    break;

                case ActivationType::SIGMOID:
                    // Sigmoid activation
                    activated_output_matrix(i, j) = Sigmoid_Function(value);
                    break;

                case ActivationType::TANH:
                    // Tanh activation
                    activated_output_matrix(i, j) = Tanh(value);
                    break;

                case ActivationType::LEAKY_RELU:
                    // Leaky ReLU activation
                    activated_output_matrix(i, j) = LeakyReLU(value);
                    break;

                case ActivationType::SWISH:
                    // Swish activation
                    activated_output_matrix(i, j) = Swish(value);
                    break;

                case ActivationType::LINEAR:
                    // Linear activation (identity function)
                    activated_output_matrix(i, j) = Linear_Activation(value);
                    break;

                default:
                    // Safe fallback if an unknown activation type is encountered
                    activated_output_matrix(i, j) = 0.0f;
                    break;
            }
        }
    }
}



void Neural_Layer::Dh_Da_Function(bool is_last_layer) {
    // Determine the activation function based on the layer type
    ActivationType activation_type;


    if (is_last_layer) {
        activation_type = layer_activation.last_layer_activation_function;
    } else {
        activation_type = layer_activation.hidden_layers_activation_function;
    }

    float value, sigmoid_swish;

    // Iterate over all elements of the activated output matrix (rows x columns)
    for (int i = 0; i < activated_output_matrix.rows(); ++i) {
        for (int j = 0; j < activated_output_matrix.columns(); ++j) {
            value = activated_output_matrix(i, j);  // Fetch the value only once per loop

            switch (activation_type) {
                case ActivationType::RELU:
                    // ReLU derivative: 1 if value > 0, else 0
                    dh_da_matrix(i, j) = (value > 0) ? 1.0f : 0.0f;
                    break;

                case ActivationType::SIGMOID:
                    // Sigmoid derivative: sigma(x) * (1 - sigma(x))
                    dh_da_matrix(i, j) = value * (1.0f - value);
                    break;

                case ActivationType::TANH:
                    // Tanh derivative: 1 - tanh^2(x)
                    dh_da_matrix(i, j) = 1.0f - value * value;
                    break;

                case ActivationType::LEAKY_RELU:
                    // Leaky ReLU derivative: 1 if value > 0, else alpha (where alpha = 0.01)
                    dh_da_matrix(i, j) = (value > 0) ? 1.0f : 0.01f;
                    break;

                case ActivationType::SWISH:
                    // Swish derivative: sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
                    sigmoid_swish = Sigmoid_Function(value);
                    dh_da_matrix(i, j) = sigmoid_swish + value * sigmoid_swish * (1.0f - sigmoid_swish);
                    break;

                case ActivationType::LINEAR:
                    // Linear derivative: 1
                    dh_da_matrix(i, j) = 1.0f;
                    break;

                default:
                    dh_da_matrix(i, j) = 0.0f;  // Safe fallback
                    break;
            }
        }
    }
}




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




static void displayLayerDetails(const Neural_Layer_Information &layers_info)

{
    const auto &neural_layers = layers_info.neural_layers;

    for (int i = 0; i < neural_layers.size(); i++) {
        std::cout << "Layer Number: " << i << std::endl;
        std::cout << "Input Matrix:" << std::endl;
        Matrix::Print(neural_layers[i].input_matrix);
        std::cout << "Weight Matrix:" << std::endl;
        Matrix::Print(neural_layers[i].weights_matrix);
        std::cout << "Bias Matrix:" << std::endl;
        Matrix::Print(neural_layers[i].bias_matrix);
        std::cout << "Weighted Sum Matrix:" << std::endl;
        Matrix::Print(neural_layers[i].weight_and_input_matrix);
        std::cout << "Weighted Sum + Bias Matrix:" << std::endl;
        Matrix::Print(neural_layers[i].weight_input_bias_matrix);
        std::cout << "Activated Output Matrix:" << std::endl;
        Matrix::Print(neural_layers[i].activated_output_matrix);
        std::cout << "dh_da Matrix:" << std::endl;
        Matrix::Print(neural_layers[i].dh_da_matrix);
        std::cout << "dC_da Matrix:" << std::endl;
        Matrix::Print(neural_layers[i].dC_da_matrix);
        std::cout << "dC_dw Matrix:" << std::endl;
        Matrix::Print(neural_layers[i].dC_dw_matrix);
        std::cout << "dC_db Matrix:" << std::endl;
        Matrix::Print(neural_layers[i].dC_db_matrix);
    }
}


Matrix BinaryThreshold(const Matrix& input, float threshold ) {
    Matrix result(input.rows(), input.columns());
    for (int i = 0; i < input.rows(); ++i) {
        for (int j = 0; j < input.columns(); ++j) {
            result(i, j) = (input(i, j) >= threshold) ? 1.0f : 0.0f;
        }
    }
    return result;
}




Neural_Layer_Information Form_Network(std::initializer_list<int> layers, Matrix inputMatrix, const Matrix& outputMatrix, ActivationType hidden_activation, ActivationType output_activation) {
    std::vector<int> layerSizes = layers;
    size_t size = layerSizes.size();
    std::cout << "Size of Neural Layer: " << size << std::endl;
    int sampleSize = inputMatrix.rows();
    std::cout << "Size of Sample: " << sampleSize << std::endl;
    int parameters = inputMatrix.columns();
    std::cout << "Parameter Size: " << parameters << std::endl;

    Matrix currentInput = std::move(inputMatrix);
    Matrix::Print(currentInput);

    std::vector<Neural_Layer> neural_layers;
    neural_layers.reserve(size);

    for (size_t i = 0; i < size; i++) {
        ActivationType current_activation = (i == size - 1) ? output_activation : hidden_activation;
        neural_layers.emplace_back(currentInput.columns(), layerSizes[i], currentInput, hidden_activation,output_activation);
        currentInput = Matrix(currentInput.rows(), layerSizes[i]);
    }

    return Neural_Layer_Information{std::move(neural_layers), std::move(layerSizes), sampleSize, outputMatrix};
}




//this function will iterate the Neural_Layer_Maker function on the number of networks I want to make
void Forward_Pass(Neural_Layer_Information &neural_layer_information) {
    auto &neural_layers = neural_layer_information.neural_layers;
    auto &layers = neural_layer_information.layers_vector;

    size_t size = layers.size();
    Matrix currentInput;
    bool is_last_layer=false;

    // Iterate over each layer
    for (size_t i = 0; i < size; i++) {
        // Compute weighted sum
        neural_layers[i].Compute_Weighted_Sum();


        if (i == size - 1) {
            neural_layers[i].Activate(neural_layers[i].layer_activation.last_layer_activation_function);
            //std::cout << "Activation used: "<< ActivationTypeToString(neural_layers[i].layer_activation.last_layer_activation_function)<< std::endl;
            is_last_layer = true;
        }
        else {
            neural_layers[i].Activate(neural_layers[i].layer_activation.hidden_layers_activation_function);

            // Prepare input for the next layer
            Matrix_Resize(currentInput, neural_layers[i].activated_output_matrix.rows(), neural_layers[i].activated_output_matrix.columns());
            currentInput = neural_layers[i].activated_output_matrix;
            //std::cout << "Activation used: "<< ActivationTypeToString(neural_layers[i].layer_activation.hidden_layers_activation_function)<< std::endl;
            neural_layers[i + 1].input_matrix = currentInput;
        }

        // Precalculate for backpropagation (should be after the correct activation)
        neural_layers[i].Dh_Da_Function(is_last_layer);

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
        Back_Propagation(neural_layer_information, mean_squared_error);
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
    Matrix binaryPredictions = BinaryThreshold(neural_layers[size-1].activated_output_matrix);
    std::cout << "Final Binary Predictions:" << std::endl;
    Matrix::Print(binaryPredictions);
}



void Matrix_Fill(Matrix& matrix, float value) {
    for (int i = 0; i < matrix.rows(); ++i) {
        for (int j = 0; j < matrix.columns(); ++j) {
            matrix(i, j) = value;
        }
    }
}
