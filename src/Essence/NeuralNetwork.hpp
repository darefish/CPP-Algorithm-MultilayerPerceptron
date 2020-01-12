#ifndef NeuralNetwork_hpp
#define NeuralNetwork_hpp

#include <vector>

#include <cmath>

#include "Neuron.hpp"

#include "Vector.hpp"

#include "Matrix.hpp"

class NeuralNetwork {
private:
    std::vector<std::vector<long double>> input;
    
    std::vector<std::vector<long double>> output;
    
    std::vector<std::vector<Neuron>> structure;
    
    void initialize();
    
    void reset();
    
    void shuffle(std::vector<std::vector<long double>>& vector);
    
    long double crossEntropyLoss(long double actualOutput,long double targetOutput);
    
    std::vector<long double> getLayerOutput(unsigned long long int layer);
    
    std::vector<long double> forwardPropagation(std::vector<long double>& input,long double(*forward)(long double));
    
    std::vector<std::vector<long double>> getLayerError(std::vector<long double>& error,long double(*backward)(long double));
    
    void backwardPropagation(long double learningRate,std::vector<std::vector<long double>>& layerError);
public:
    NeuralNetwork(std::vector<unsigned long long int> structure,std::vector<std::vector<long double>>& input,std::vector<std::vector<long double>>& output);
    
    long double getLoss(std::vector<long double> actualOutput,std::vector<long double> targetOutput);
    
    long double getCost(long double(*forward)(long double));
    
    std::vector<long double> predict(std::vector<long double> input,long double(*forward)(long double));
    
    void batchGradientDescent(unsigned long long int iterations,long double learningRate,long double(*forward)(long double),long double(*backward)(long double));
    
    void stochasticGradientDescent(unsigned long long int iterations,long double learningRate,long double(*forward)(long double),long double(*backward)(long double));
    
    void miniBatchGradientDescent(unsigned long long int batchSize,unsigned long long int iterations,long double learningRate,long double(*forward)(long double),long double(*backward)(long double));
    
    ~NeuralNetwork();
};

#endif
