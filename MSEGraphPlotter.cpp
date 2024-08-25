#include "MSEGraphPlotter.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>

void MSEGraphPlotter::plotMSEvsIterations(const std::vector<float>& mseValues, const std::vector<int>& iterations) {
    const int width = 80;
    const int height = 20;
    const int leftMargin = 9;

    float minMSE = *std::min_element(mseValues.begin(), mseValues.end());
    float maxMSE = *std::max_element(mseValues.begin(), mseValues.end());
    int maxIterations = iterations.back();

    std::cout << std::string(width, '-') << std::endl;
    std::cout << "Mean Squared Error vs Iterations" << std::endl;
    std::cout << std::string(width, '-') << std::endl;

    std::vector<std::vector<char>> graph(height, std::vector<char>(width - leftMargin, ' '));
    plotPoints(graph, width - leftMargin, height, mseValues, iterations, minMSE, maxMSE, maxIterations);

    for (int i = height - 1; i >= 0; --i) {  // Reversed loop to fix the Y-axis orientation
        float mseValue = minMSE + (maxMSE - minMSE) * i / (height - 1);  // Correct the formula for MSE calculation
        std::cout << std::setw(8) << formatFloat(mseValue, 2) << " |";
        for (int j = 0; j < width - leftMargin; ++j) {
            std::cout << graph[i][j];
        }
        std::cout << std::endl;
    }

    std::cout << std::string(leftMargin, '-') << "|" << std::string(width - leftMargin - 1, '-') << std::endl;
    std::cout << std::setw(leftMargin) << "0" << std::setw(width - leftMargin) << maxIterations << std::endl;
    std::cout << std::setw(width / 2) << "Iterations" << std::endl;

    std::cout << std::string(width, '-') << std::endl;
    std::cout << "Legend: * - MSE value" << std::endl;
    std::cout << std::string(width, '-') << std::endl;
}

void MSEGraphPlotter::plotPoints(std::vector<std::vector<char>>& graph, int width, int height,
                                 const std::vector<float>& mseValues, const std::vector<int>& iterations,
                                 float minMSE, float maxMSE, int maxIterations) {
    for (size_t i = 0; i < mseValues.size(); ++i) {
        int x = mapToX(iterations[i], maxIterations, width);
        int y = mapToY(mseValues[i], minMSE, maxMSE, height);
        if (x >= 0 && x < width && y >= 0 && y < height) {
            graph[y][x] = '*';
        }
    }
}

int MSEGraphPlotter::mapToY(float value, float minValue, float maxValue, int height) {
    // Fixed mapping so that higher MSE values are at the top
    return static_cast<int>((height - 1) * ((value - minValue) / (maxValue - minValue)));
}

int MSEGraphPlotter::mapToX(int iteration, int maxIteration, int width) {
    return static_cast<int>(static_cast<float>(iteration) / maxIteration * (width - 1));
}

std::string MSEGraphPlotter::formatFloat(float value, int precision) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << value;
    return oss.str();
}
