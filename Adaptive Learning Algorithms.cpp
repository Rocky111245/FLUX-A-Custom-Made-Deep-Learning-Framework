//
// Created by rakib on 29/4/2024.
//

#include "Adaptive Learning Algorithms.h"
#include <vector>


void NAG_First_Call(std::tuple<std::vector<Neural_Layer>, std::vector<Matrix>, std::vector<Matrix>>& network_data,int layer_number,float momentum){

    auto &[neural_layers,velocity,temp_modified_velocity]=network_data;


    Matrix_Fill(temp_modified_velocity[layer_number],0);
    //Makes a deep copy of the current layer's velocity
    Matrix_DeepCopy(temp_modified_velocity[layer_number],velocity[layer_number]);
    //scales the matrix of that specific layer with the momentum
    Matrix_Scalar_Multiply(temp_modified_velocity[layer_number],momentum);

    //Creation of temporary weights matrix
    Matrix temporary_weight= Matrix_Create_Zero(neural_layers[layer_number].weights_matrix.row,neural_layers[layer_number].weights_matrix.column);
    // Modified the weight. This modified weight will be used to find the dC_dW
    Matrix_Subtract(&temporary_weight,neural_layers[layer_number].weights_matrix,temp_modified_velocity[layer_number]);

//  Compute dC_da using the modified weights
    Matrix n = Matrix_Create_Zero(neural_layers[layer_number].weights_matrix.column, neural_layers[layer_number].weights_matrix.row);
    Matrix_Transpose_v2(&n, temporary_weight);

    if (layer_number != 0) {
        Matrix temp_delta = Matrix_Create_Zero(neural_layers[layer_number].dC_da_matrix.row, n.column);
        Matrix_Multiply(&temp_delta, neural_layers[layer_number].dC_da_matrix, n);

        Matrix_Hadamard_Product(neural_layers[layer_number - 1].dC_da_matrix, temp_delta, neural_layers[layer_number - 1].dh_da_matrix);
        Matrix_Free(&temp_delta);
    }
    Matrix_Free(&n);

//  find dC_dw for each layer using the modified weights
    Matrix m = Matrix_Create_Zero(neural_layers[layer_number].input_matrix.column, neural_layers[layer_number].input_matrix.row);
    Matrix_Transpose_v2(&m, neural_layers[layer_number].input_matrix);

    Matrix_Multiply(&neural_layers[layer_number].dC_dw_matrix, m, neural_layers[layer_number].dC_da_matrix);
    Matrix_Free(&m);

    // Averaging the gradients by the batch size (assuming batch size is 4)
    Matrix_Scalar_Multiply(neural_layers[layer_number].dC_dw_matrix, 1.0f / 4.0f);

    Matrix_Sum_Columns(neural_layers[layer_number].dC_db_matrix, neural_layers[layer_number].dC_da_matrix);

    // Averaging dC_db for the batch
    Matrix_Scalar_Multiply(neural_layers[layer_number].dC_db_matrix, 1.0f / 4.0f);
    Matrix_Free(&temporary_weight);


}


void NAG_Second_Call(std::tuple<std::vector<Neural_Layer>, std::vector<Matrix>, std::vector<Matrix>>& network_data, size_t layer_number){
    auto &[neural_layers,velocity,temp_modified_velocity]=network_data;

    //  Update the velocity. This will happen in another function called the learn function where dC_dw has already been modified by the learning rate
    Matrix_Add(&velocity[layer_number],temp_modified_velocity[layer_number],neural_layers[layer_number].dC_dw_matrix);
    Matrix_Subtract(&neural_layers[layer_number].weights_matrix,neural_layers[layer_number].weights_matrix,velocity[layer_number]);
}












//// Initialize velocities for each layer

//for each layer in layers:
//velocity[layer] = 0

//

//// Iterate over epochs
//for each epoch:
//// Iterate over each layer
//for each layer in layers:
//// Step 1: Temporary parameter update using lookahead based on current momentum
//temp_weights[layer] = weights[layer] - momentum * velocity[layer]
//
//// Step 2: Compute gradient at temporary lookahead weights
//grad[layer] = compute_gradient(temp_weights[layer])
//
//// Step 3: Update velocity based on gradient at lookahead position
//velocity[layer] = momentum * velocity[layer] + learning_rate * grad[layer]
//
//// Step 4: Apply the updated velocity to the actual weights
//weights[layer] = weights[layer] - velocity[layer]
