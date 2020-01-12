#include <iostream>

#include <ctime>

#include "File.hpp"

#include "Timer.hpp"

#include "NeuralNetwork.hpp"

#include "ActivationFunction.hpp"

std::ostream& operator<<(std::ostream& ostream,std::vector<long double>& vector) {
    ostream<<"[";
    
    for(auto i=0;i<vector.size();++i) {
        ostream<<vector[i];
        
        if(i!=vector.size()-1) {
            ostream<<",";
        }
    }
    
    ostream<<"]";
    
    return ostream;
}

std::ostream& operator<<(std::ostream& ostream,Timer& timer) {
    ostream<<timer.process<<" took "<<timer.duration.count()*1000.0<<" ms";
    
    return ostream;
}

std::vector<std::vector<long double>> toMatrix(std::vector<long double> vector) {
    std::vector<std::vector<long double>> output;
    
    for(auto i=0;i<vector.size();++i) {
        std::vector<long double> local;
        
        for(auto x=0;x<10;++x) {
            if(x==vector[i]) {
                local.push_back(1);
            }else {
                local.push_back(0);
            }
        }
        output.push_back(local);
    }
    
    return output;
}

int main() {
    
    srand((unsigned)time(nullptr));
    
    NeuralNetwork* nn;
    
    std::vector<std::vector<long double>> input;

    std::vector<std::vector<long double>> output;

    {
        Timer timer("Data Initialization");

        timer.start();

        input=File::readAsMatrix("/Users/jiangtengda/Desktop/XOR/input.txt",2);

        output=File::readAsMatrix("/Users/jiangtengda/Desktop/XOR/output.txt",1);

        timer.stop();

        std::cout<<timer<<std::endl;
    }

    {
        Timer timer("NeuralNetwork Initialization");

        timer.start();

        nn=new NeuralNetwork({2,2,1},input,output);

        timer.stop();

        std::cout<<timer<<std::endl;
    }
    
    std::cout<<"Before Training"<<std::endl;

    std::cout<<nn->getCost(ActivationFunction::hyperbolicTangent)<<std::endl;

    for(auto i=0;i<input.size();++i) {
        std::vector<long double> actualOutput=nn->predict(input[i],ActivationFunction::hyperbolicTangent);
        std::cout<<input[i]<<" -> "<<actualOutput<<std::endl;
    }

    {
        Timer timer("NeuralNetwork Training Time");

        timer.start();

        nn->stochasticGradientDescent(10000,0.5,ActivationFunction::hyperbolicTangent,ActivationFunction::derivativeHyperbolicTangent);

        timer.stop();

        std::cout<<timer<<std::endl;
    }

    std::cout<<"After Training"<<std::endl;

    std::cout<<nn->getCost(ActivationFunction::hyperbolicTangent)<<std::endl;

    for(auto i=0;i<input.size();++i) {
        std::vector<long double> actualOutput=nn->predict(input[i],ActivationFunction::hyperbolicTangent);
        std::cout<<input[i]<<" -> "<<actualOutput<<std::endl;
    }
}
