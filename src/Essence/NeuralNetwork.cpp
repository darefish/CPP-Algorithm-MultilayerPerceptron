#include "NeuralNetwork.hpp"

void NeuralNetwork::initialize() {
    for(auto i=0;i<structure.size()-1;++i) {
        for(auto x=0;x<structure[i].size();++x) {
            for(auto y=0;y<structure[i+1].size();++y) {
                structure[i][x].connect(&structure[i+1][y]);
            }
        }
    }
}

void NeuralNetwork::reset() {
    for(auto x=0;x<structure.size();++x) {
        for(auto y=0;y<structure[x].size();++y) {
            structure[x][y].value=0.0;
        }
    }
}

void NeuralNetwork::shuffle(std::vector<std::vector<long double>>& vector) {
    for(auto i=0;i<vector.size()/2;++i) {
        unsigned long long int one=rand()%vector.size();
        unsigned long long int two=rand()%vector.size();
        
        std::swap(vector[one],vector[two]);
    }
}

long double NeuralNetwork::crossEntropyLoss(long double actualOutput,long double targetOutput) {
    if(targetOutput==1) {
        return -log(actualOutput);
    }else {
        return -log(1-actualOutput);
    }
}

std::vector<long double> NeuralNetwork::getLayerOutput(unsigned long long int layer) {
    std::vector<long double> output;
    
    for(auto i=0;i<structure[layer].size();++i) {
        output.push_back(structure[layer][i].value);
    }
    return output;
}

std::vector<long double> NeuralNetwork::forwardPropagation(std::vector<long double>& input,long double(*forward)(long double)) {
    
    for(auto i=0;i<structure[0].size();++i) {
        structure[0][i].value+=input[i];
        structure[0][i].activate();
    }
    
    for(auto i=1;i<structure.size();++i) {
        for(auto x=0;x<structure[i].size();++x) {
            structure[i][x].value+=structure[i][x].bias;
            structure[i][x].value=(*forward)(structure[i][x].value);
            structure[i][x].activate();
        }
    }
    
    return getLayerOutput(structure.size()-1);
}

std::vector<std::vector<long double>> NeuralNetwork::getLayerError(std::vector<long double>& error,long double(*backward)(long double)) {

    std::vector<std::vector<long double>> layerError;
    
    std::vector<long double> localOutput=getLayerOutput(structure.size()-1);
    std::vector<long double> localDerivative=Vector::operationOfVector(localOutput,backward);
    std::vector<long double> localError=Vector::multiplicationOfVector(localDerivative,error);
    
    layerError.push_back(localError);
    
    for(long int i=structure.size()-2;i>0;i--) {
        
        std::vector<long double> localError;
        
        for(auto x=0;x<structure[i].size();++x) {
            std::vector<long double> localSum=Vector::multiplicationOfVector(structure[i][x].weights,layerError[structure.size()-2-i]);
            localError.push_back(Vector::sumOfVector(localSum));
        }
        
        std::vector<long double> localOutput=getLayerOutput(i);
        
        std::vector<long double> localDerivative=Vector::operationOfVector(localOutput,backward);
        
        localError=Vector::multiplicationOfVector(localDerivative,localError);
        
        layerError.push_back(localError);
    }
    
    return layerError;
}

void NeuralNetwork::backwardPropagation(long double learningRate,std::vector<std::vector<long double>>& layerError) {
    for(auto i=0;i<structure.size()-1;++i) {
        
        for(auto x=0;x<structure[i].size();++x) {
            for(auto y=0;y<structure[i][x].weights.size();++y) {
                structure[i][x].weights[y]-=layerError[layerError.size()-1-i][y]*structure[i][x].value*learningRate;
            }
        }
        
        for(auto x=0;x<structure[i+1].size();++x) {
            structure[i+1][x].bias-=layerError[layerError.size()-i-1][x]*learningRate;
        }
        
    }
}

NeuralNetwork::NeuralNetwork(std::vector<unsigned long long int> structure,std::vector<std::vector<long double>>& input,std::vector<std::vector<long double>>& output) {
    for(auto i=0;i<structure.size();++i) {
        this->structure.push_back(std::vector<Neuron>(structure[i]));
    }
    
    this->input=input;
    this->output=output;
    
    initialize();
}

long double NeuralNetwork::getLoss(std::vector<long double> actualOutput,std::vector<long double> targetOutput) {
    long double value=0.0;
    
    for(auto i=0;i<actualOutput.size();++i) {
        value+=crossEntropyLoss(actualOutput[i],targetOutput[i]);
    }
    
    return value;
}

long double NeuralNetwork::getCost(long double(*forward)(long double)) {
    long double value=0.0;
    
    for(auto i=0;i<input.size();++i) {
        value+=getLoss(predict(input[i],forward),output[i]);
    }
    
    value/=input.size();
    
    return value;
}

void NeuralNetwork::batchGradientDescent(unsigned long long int iterations,long double learningRate,long double(*forward)(long double),long double(*backward)(long double)) {
    for(unsigned long long int i=0;i<iterations;++i) {
        
        std::vector<std::vector<std::vector<long double>>> gradients;
        
        for(unsigned long long int e=0;e<input.size();++e) {
            std::vector<long double> actualOutput=forwardPropagation(input[e],forward);
            std::vector<long double> error=Vector::subtractionOfVector(actualOutput,output[e]);
            std::vector<std::vector<long double>> layerError=getLayerError(error,backward);
            
            gradients.push_back(layerError);
            
            reset();
        }
        
        std::vector<std::vector<long double>> averageLayerError=Matrix::averageOfMatrix(gradients);
        
        backwardPropagation(learningRate,averageLayerError);
    }
}

void NeuralNetwork::stochasticGradientDescent(unsigned long long int iterations,long double learningRate,long double(*forward)(long double),long double(*backward)(long double)) {
    for(unsigned long long int i=0;i<iterations;++i) {
        
        for(unsigned long long int e=0;e<input.size();++e) {
            unsigned long long int index=rand()%input.size();
        
            std::vector<long double> actualOutput=forwardPropagation(input[index],forward);
        
            std::vector<long double> error=Vector::subtractionOfVector(actualOutput,output[index]);
        
            std::vector<std::vector<long double>> layerError=getLayerError(error,backward);
        
            backwardPropagation(learningRate,layerError);
        
            reset();
        }
    }
}

void NeuralNetwork::miniBatchGradientDescent(unsigned long long int batchSize,unsigned long long int iterations,long double learningRate,long double(*forward)(long double),long double(*backward)(long double)) {
    for(unsigned long long int i=0;i<iterations;++i) {
        
        for(unsigned long long int e=0;e<input.size()/batchSize;++e) {
            
            std::vector<std::vector<std::vector<long double>>> gradients;
            
            for(unsigned long long int x=0;x<batchSize;++x) {
                unsigned long long int index=e*batchSize+x;
                
                std::vector<long double> actualOutput=forwardPropagation(input[index],forward);
                std::vector<long double> error=Vector::subtractionOfVector(actualOutput,output[index]);
                std::vector<std::vector<long double>> layerError=getLayerError(error,backward);
                
                gradients.push_back(layerError);
                
                reset();
            }
            
            std::vector<std::vector<long double>> averageLayerError=Matrix::averageOfMatrix(gradients);
            
            backwardPropagation(learningRate,averageLayerError);
        }
        
        std::vector<std::vector<std::vector<long double>>> gradients;
        
        for(unsigned long long int e=0;e<input.size()%batchSize;++e) {
            unsigned long long int index=input.size()%batchSize+e;
            
            std::vector<long double> actualOutput=forwardPropagation(input[index],forward);
            std::vector<long double> error=Vector::subtractionOfVector(actualOutput,output[index]);
            std::vector<std::vector<long double>> layerError=getLayerError(error,backward);
            
            gradients.push_back(layerError);
            
            reset();
        }
        
        if(gradients.size()>0) {
        
        std::vector<std::vector<long double>> averageLayerError=Matrix::averageOfMatrix(gradients);
        
        backwardPropagation(learningRate,averageLayerError);
        }
        
        shuffle(input);
    }
}

std::vector<long double> NeuralNetwork::predict(std::vector<long double> input,long double(*function)(long double)) {
    std::vector<long double> output=forwardPropagation(input,function);
    reset();
    return output;
}

NeuralNetwork::~NeuralNetwork() {
    
}
