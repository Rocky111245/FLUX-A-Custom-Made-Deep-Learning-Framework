#ifndef _DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_MSEGRAPHPLOTTER_H
#define _DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_MSEGRAPHPLOTTER_H

#include <vector>
#include <string>

class MSEGraphPlotter {
public:
    static void plotMSEvsIterations(const std::vector<float>& mseValues, const std::vector<int>& iterations);

private:
    static void plotPoints(std::vector<std::vector<char>>& graph, int width, int height,
                           const std::vector<float>& mseValues, const std::vector<int>& iterations,
                           float minMSE, float maxMSE, int maxIterations);
    static int mapToY(float value, float minValue, float maxValue, int height);
    static int mapToX(int iteration, int maxIteration, int width);
    static std::string formatFloat(float value, int precision);
};

#endif // _DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_MSEGRAPHPLOTTER_H
