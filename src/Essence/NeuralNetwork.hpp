#ifndef NeuralNetwork_hpp
#define NeuralNetwork_hpp

#include <vector>

#include <cmath>

#include "File.hpp"

#include "Neuron.hpp"

#include "Vector.hpp"

#include "Matrix.hpp"

class NeuralNetwork {
private:
    std::vector<std::vector<Neuron>> structure;
    
    std::vector<std::vector<long double>> input;
    
    std::vector<std::vector<long double>> output;
    
    long double(*forward)(long double);
    
    long double(*backward)(long double);
    
    void initialize();
    
    void reset();
    
    void shuffle(std::vector<std::vector<long double>>& input,std::vector<std::vector<long double>>& output);
    
    long double crossEntropyLoss(long double actualOutput,long double targetOutput);
    
    std::vector<long double> getLayerOutput(unsigned long long int layer);
    
    std::vector<long double> forwardPropagation(std::vector<long double>& input);
    
    std::vector<std::vector<long double>> getBiasesLayerError(std::vector<long double>& actualOutput,std::vector<long double>& targetOutput);
    
    std::vector<std::vector<long double>> getWeightsLayerError(std::vector<std::vector<long double>>& biasesLayerError);
    
    void backwardPropagation(long double learningRate,std::vector<std::vector<long double>>& biasesLayerError,std::vector<std::vector<long double>>& weightsLayerError);
public:
    NeuralNetwork(std::vector<unsigned long long int> structure);
    
    void setActivationFunction(long double(*forward)(long double),long double(*backward)(long double));
    
    void loadData(std::vector<std::vector<long double>>& input,std::vector<std::vector<long double>>& output);
    
    long double getLoss(std::vector<long double> actualOutput,std::vector<long double> targetOutput);
    
    long double getCost();
    
    std::vector<long double> predict(std::vector<long double> input);
    
    void batchGradientDescent(unsigned long long int iterations,long double learningRate,bool track,const char* directory);
    
    void stochasticGradientDescent(unsigned long long int iterations,long double learningRate,bool track,const char* directory);
    
    void miniBatchGradientDescent(unsigned long long int batchSize,unsigned long long int iterations,long double learningRate,bool track,const char* directory);
    
    ~NeuralNetwork();
};

#endif
