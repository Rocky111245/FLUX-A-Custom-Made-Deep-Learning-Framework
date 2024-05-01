#include <iostream>
#include <vector>
#include "Neural Network Framework.h"
#include "Excel.h"



int main() {
    try {
        // The full path to the CSV file including the filenameInput and its extension
        std::string filenameInput = R"(D:\Software\Machine Learning\Datasets\redWine.csv)";
        int num_rows_input = 600;  // The number of rows to read, excluding the header
        int num_columns_input = 11;  // The number of columns to read
        std::string filenameOutput = R"(D:\Software\Machine Learning\Datasets\redWineOutput.csv)";
        int num_rows_output = 600;  // The number of rows to read, excluding the header
        int num_columns_output = 1;  // The number of columns to read

        // Call the function to read the CSV and convert it to a matrix
       Matrix input=From_CSV_To_Matrix(filenameInput, num_rows_input, num_columns_input);
       Matrix output=From_CSV_To_Matrix(filenameOutput, num_rows_output, num_columns_output);

       std::vector<Neural_Layer> network = Form_Network({15,1}, input);
       Learn(network,{15,1},output,0.00012,400000);



    } catch (const std::runtime_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1; // Return an error code
    }

    return 0; // Return success
}


//Under Test environment


//    Matrix inputs= Matrix_Maker_2DArray(4, 8, 3, 0, &data[0][0]);
//    Matrix_Display(inputs);
//
//    Matrix output= Matrix_Maker_2DArray(4,8,1,3,&data[0][0]);
//    Matrix_Display(output);
//
//

//    std::vector<Neural_Layer> network = Form_Network({3,3,1}, inputs);
//    Learn(network,{3,3,1},inputs,output,0.0014,405000);