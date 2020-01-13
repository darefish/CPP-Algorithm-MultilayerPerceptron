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

void NeuralNetwork::shuffle(std::vector<std::vector<long double>>& input,std::vector<std::vector<long double>>& output) {
    for(auto i=0;i<input.size()/2;++i) {
        unsigned long long int one=rand()%input.size();
        unsigned long long int two=rand()%input.size();
        
        std::swap(input[one],input[two]);
        std::swap(output[one],output[two]);
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

std::vector<long double> NeuralNetwork::forwardPropagation(std::vector<long double>& input) {
    
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

std::vector<std::vector<long double>> NeuralNetwork::getBiasesLayerError(std::vector<long double>& actualOutput,std::vector<long double>& targetOutput) {

    std::vector<std::vector<long double>> biasesLayerError;
    
    std::vector<long double> difference=Vector::subtractionOfVector(actualOutput,targetOutput);
    
    std::vector<long double> output=getLayerOutput(structure.size()-1);
    
    std::vector<long double> derivative=Vector::operationOfVector(output,backward);
    
    std::vector<long double> error=Vector::multiplicationOfVector(difference,derivative);
    
    biasesLayerError.push_back(error);
    
    for(auto i=structure.size()-2;i>0;--i) {
        std::vector<long double> difference;
        
        std::vector<long double> output=getLayerOutput(i);
        
        std::vector<long double> derivative=Vector::operationOfVector(output,backward);
        
        for(auto x=0;x<structure[i].size();++x) {
            std::vector<long double> sum=Vector::multiplicationOfVector(structure[i][x].weights,biasesLayerError[structure.size()-2-i]);
            difference.push_back(Vector::sumOfVector(sum));
        }
        
        std::vector<long double> error=Vector::multiplicationOfVector(derivative,difference);
        
        biasesLayerError.push_back(error);
    }
    
    return biasesLayerError;
}

std::vector<std::vector<long double>> NeuralNetwork::getWeightsLayerError(std::vector<std::vector<long double>>& biasesLayerError) {
    
    std::vector<std::vector<long double>> weightsLayerError;
    
    for(auto i=0;i<biasesLayerError.size();++i) {
        std::vector<Neuron> layerValue=structure[structure.size()-2-i];
        
        std::vector<long double> local;
        
        for(auto x=0;x<layerValue.size();++x) {
            for(auto y=0;y<biasesLayerError[i].size();++y) {
                local.push_back(layerValue[x].value*biasesLayerError[i][y]);
            }
        }
        weightsLayerError.push_back(local);
    }
    
    return weightsLayerError;
}

void NeuralNetwork::backwardPropagation(long double learningRate,std::vector<std::vector<long double>>& biasesLayerError,std::vector<std::vector<long double>>& weightsLayerError) {
    for(auto i=0;i<biasesLayerError.size();++i) {
        for(auto x=0;x<structure[structure.size()-1-i].size();++x) {
            structure[structure.size()-1-i][x].bias-=biasesLayerError[i][x]*learningRate;
        }
    }
    
    for(auto i=0;i<weightsLayerError.size();++i) {
        for(auto x=0;x<structure[structure.size()-2-i].size();++x) {
            for(auto y=0;y<structure[structure.size()-2-i][x].weights.size();++y) {
                structure[structure.size()-2-i][x].weights[y]-=weightsLayerError[i][x*structure[structure.size()-2-i][x].weights.size()+y]*learningRate;
            }
        }
    }
}

NeuralNetwork::NeuralNetwork(std::vector<unsigned long long int> structure) {
    for(auto i=0;i<structure.size();++i) {
        this->structure.push_back(std::vector<Neuron>(structure[i]));
    }
    initialize();
}

void NeuralNetwork::setActivationFunction(long double(*forward)(long double),long double(*backward)(long double)) {
    this->forward=forward;
    this->backward=backward;
}

void NeuralNetwork::loadData(std::vector<std::vector<long double>>& input,std::vector<std::vector<long double>>& output) {
    this->input=input;
    this->output=output;
}

long double NeuralNetwork::getLoss(std::vector<long double> actualOutput,std::vector<long double> targetOutput) {
    long double value=0.0;
    
    for(auto i=0;i<actualOutput.size();++i) {
        value+=crossEntropyLoss(actualOutput[i],targetOutput[i]);
    }
    
    return value;
}

long double NeuralNetwork::getCost() {
    long double value=0.0;
    
    for(auto i=0;i<input.size();++i) {
        value+=getLoss(predict(input[i]),output[i]);
    }
    
    value/=input.size();
    
    return value;
}

void NeuralNetwork::batchGradientDescent(unsigned long long int iterations,long double learningRate,bool track,const char* directory) {
    std::vector<long double> cost;
    
    for(unsigned long long int i=0;i<iterations;++i) {
        
        std::vector<std::vector<std::vector<long double>>> biasesGradients;
        
        std::vector<std::vector<std::vector<long double>>> weightsGradients;
        
        for(unsigned long long int e=0;e<input.size();++e) {
            std::vector<long double> actualOutput=forwardPropagation(input[e]);
            
            std::vector<std::vector<long double>> biasesLayerError=getBiasesLayerError(actualOutput,output[e]);
            
            std::vector<std::vector<long double>> weightsLayerError=getWeightsLayerError(biasesLayerError);
            
            biasesGradients.push_back(biasesLayerError);
            
            weightsGradients.push_back(weightsLayerError);
            
            reset();
        }
        
        std::vector<std::vector<long double>> averageBiasesGradients=Matrix::averageOfMatrix(biasesGradients);
        
        std::vector<std::vector<long double>> averageWeightsGradients=Matrix::averageOfMatrix(weightsGradients);
        
        backwardPropagation(learningRate,averageBiasesGradients,averageWeightsGradients);
        
        cost.push_back(getCost());
    }
    if(track) {
        File::writeAsVector(directory,cost);
    }
}

void NeuralNetwork::stochasticGradientDescent(unsigned long long int iterations,long double learningRate,bool track,const char* directory) {
    std::vector<long double> cost;
    
    for(unsigned long long int i=0;i<iterations;++i) {
        
        for(unsigned long long int e=0;e<input.size();++e) {
        
            std::vector<long double> actualOutput=forwardPropagation(input[e]);
        
            std::vector<std::vector<long double>> biasesLayerError=getBiasesLayerError(actualOutput,output[e]);
        
            std::vector<std::vector<long double>> weightsLayerError=getWeightsLayerError(biasesLayerError);
            
            backwardPropagation(learningRate,biasesLayerError,weightsLayerError);
            
            reset();
            
            cost.push_back(getCost());
        }
        shuffle(input,output);
    }
    if(track) {
        File::writeAsVector(directory,cost);
    }
}

void NeuralNetwork::miniBatchGradientDescent(unsigned long long int batchSize,unsigned long long int iterations,long double learningRate,bool track,const char* directory) {
    std::vector<long double> cost;
    
    for(unsigned long long int i=0;i<iterations;++i) {
        
        for(unsigned long long int e=0;e<input.size()/batchSize;++e) {
            
            std::vector<std::vector<std::vector<long double>>> biasesGradients;
            
            std::vector<std::vector<std::vector<long double>>> weightsGradients;
            
            for(unsigned long long int b=0;b<batchSize;++b) {
                std::vector<long double> actualOutput=forwardPropagation(input[e*batchSize+b]);
                
                std::vector<std::vector<long double>> biasesLayerError=getBiasesLayerError(actualOutput,output[e*batchSize+b]);
                
                std::vector<std::vector<long double>> weightsLayerError=getWeightsLayerError(biasesLayerError);
                
                biasesGradients.push_back(biasesLayerError);
                
                weightsGradients.push_back(weightsLayerError);
                
                reset();
            }
            
            std::vector<std::vector<long double>> averageBiasesGradients=Matrix::averageOfMatrix(biasesGradients);
            
            std::vector<std::vector<long double>> averageWeightsGradients=Matrix::averageOfMatrix(weightsGradients);
            
            backwardPropagation(learningRate,averageBiasesGradients,averageWeightsGradients);
            
            cost.push_back(getCost());
        }
    }
    
    if(track) {
        File::writeAsVector(directory,cost);
    }
}

std::vector<long double> NeuralNetwork::predict(std::vector<long double> input) {
    std::vector<long double> output=forwardPropagation(input);
    reset();
    return output;
}

NeuralNetwork::~NeuralNetwork() {
    
}
