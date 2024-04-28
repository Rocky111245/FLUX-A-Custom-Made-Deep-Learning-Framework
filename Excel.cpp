//
// Created by rakib on 27/4/2024.
//
#include "Excel.h"
#include "Helper.h"


// Function to read CSV file and convert to float array
float* Read_Csv_to_Array(const std::string& filename, int num_rows, int num_columns) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file");
    }

    // Allocate memory for the float array
    auto* data = new float[num_rows * num_columns];
    std::string line;
    std::string cell;

    // Skip header row
    std::getline(file, line);

    // Read the specified number of rows
    for (int i = 0; i < num_rows && std::getline(file, line); ++i) {
        std::stringstream linestream(line);

        // Read the specified number of columns
        for (int j = 0; j < num_columns && std::getline(linestream, cell, ';'); ++j) {
            // Convert string to float and store in array
            data[i * num_columns + j] = std::stof(cell);
        }
    }

    file.close();
    return data;
}

// Function to read fixed-width file and convert to float array
float* Read_Fixed_Width_to_Array(const std::string& filename, int num_rows, int num_columns, int column_width) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file");
    }

    auto* data = new float[num_rows * num_columns];
    std::string line;

    // Skip header row by reading and discarding it
    std::getline(file, line);

    // Start reading from the second row
    for (int i = 0; i < num_rows && std::getline(file, line); ++i) {
        int start = 0; // Initialize start position for the first column
        for (int j = 0; j < num_columns; ++j) {
            // Extract the substring for the current column
            std::string cell = line.substr(start, column_width);
            start += column_width; // Move start to the next column

            // Trim whitespace from the cell
            cell.erase(0, cell.find_first_not_of(" \t\n\r\f\v"));
            cell.erase(cell.find_last_not_of(" \t\n\r\f\v") + 1);

            // Convert string to float and store in array
            try {
                data[i * num_columns + j] = std::stof(cell);
            } catch (const std::invalid_argument& e) {
                // Handle the case where the string is not a valid float
                std::cerr << "Invalid argument for stof conversion: " << cell << std::endl;
                // Set to zero or some other default value if needed
                data[i * num_columns + j] = 0.0f;
            }
        }
    }

    file.close();
    return data;
}



void Normalize_Data(Matrix& matrix) {
    for (int j = 0; j < matrix.column; ++j) {  // Iterate over each column
        float min_value = std::numeric_limits<float>::max();
        float max_value = std::numeric_limits<float>::lowest();

        // Find min and max values for this column across all rows
        for (int i = 0; i < matrix.row; ++i) {
            float value = matrix.data[i * matrix.column + j]; // Access the element at the ith row and jth column
            min_value = std::min(min_value, value); // Update min_value if the current value is smaller
            max_value = std::max(max_value, value); // Update max_value if the current value is larger
        }

        // Normalize each element in this column
        for (int i = 0; i < matrix.row; ++i) {
            int index = i * matrix.column + j; // Calculate the index again for the ith row and jth column
            // Apply normalization formula: (current value - min) / (max - min)
            matrix.data[index] = (matrix.data[index] - min_value) / (max_value - min_value);
        }
    }
}



// Function to copy data from source matrix to destination matrix.
void Matrix_Create_From_CSV(Matrix &destination, const float *source) {
    size_t size=destination.column*destination.row;
    for (int i=0;i<size;i++){
        destination.data[i]=source[i];
    }
}


Matrix From_CSV_To_Matrix(const std::string& filename, int num_rows, int num_columns){
    Matrix A= Matrix_Create_Zero(num_rows,num_columns);
    float *data= Read_Csv_to_Array(filename, num_rows, num_columns);
    Matrix_Create_From_CSV(A,data);
    std::cout<<"Before Normalization"<<std::endl;
//    Matrix_Display(A);
    Normalize_Data(A);
    std::cout<<"After Normalization"<<std::endl;
//    Matrix_Display(A);
    return A;
}

void From_CSV_Width_to_Matrix(const std::string& filename, int num_rows, int num_columns, int column_width){
    Matrix A= Matrix_Create_Zero(num_rows,num_columns);
    float *data= Read_Fixed_Width_to_Array(filename, num_rows, num_columns, column_width);
    //normalize here
    Matrix_Create_From_CSV(A,data);
   // Matrix_Display(A);
}
