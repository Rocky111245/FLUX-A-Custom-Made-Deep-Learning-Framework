cmake_minimum_required(VERSION 3.26)
project(_Discriminative_Dense_Neural_Network_Framework)

set(CMAKE_CXX_STANDARD 23)

# Add an executable target called _Discriminative_Dense_Neural_Network_Framework to the project using the specified source file
add_executable(_Discriminative_Dense_Neural_Network_Framework
        main.cpp
        "Neural Network Framework.cpp"
        "Neural Network Framework.h"
        "MSEGraphPlotter.cpp"
        "MSEGraphPlotter.h"
        "NNStructureVis .cpp"
        "NNStructureVis .h"
)

# Specify the directory where the header file can be found
target_include_directories(_Discriminative_Dense_Neural_Network_Framework PRIVATE "D:/Github Files/C++_Matrix_Library_for_Neural_Networks")

# Specify the directory where the .dll and .lib files can be found for linking
# Assuming the .lib file is located in the cmake-build-debug directory or its "lib" subdirectory
target_link_directories(_Discriminative_Dense_Neural_Network_Framework PRIVATE "D:/Github Files/C++_Matrix_Library_for_Neural_Networks/cmake-build-debug")
target_link_libraries(_Discriminative_Dense_Neural_Network_Framework PRIVATE libMatrixLibrary)

# Copy the DLL to the executable directory after building
add_custom_command(TARGET _Discriminative_Dense_Neural_Network_Framework POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "D:/Github Files/C++_Matrix_Library_for_Neural_Networks/cmake-build-debug/libMatrixLibrary.dll"
        $<TARGET_FILE_DIR:_Discriminative_Dense_Neural_Network_Framework>
)
