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

int main() {
    srand((unsigned)time(nullptr));
    
    NeuralNetwork* nn;
    
    long double(*forward)(long double)=ActivationFunction::hyperbolicTangent;
    
    long double(*backward)(long double)=ActivationFunction::derivativeHyperbolicTangent;
    
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

        nn=new NeuralNetwork({2,2,1});
        
        nn->setActivationFunction(forward,backward);
        
        nn->loadData(input,output);

        timer.stop();

        std::cout<<timer<<std::endl;
    }
    
    std::cout<<"Before Training"<<std::endl;

    std::cout<<"Cost: "<<nn->getCost()<<std::endl;

    for(auto i=0;i<input.size();++i) {
        std::vector<long double> actualOutput=nn->predict(input[i]);
        std::cout<<input[i]<<" -> "<<actualOutput<<std::endl;
    }

    {
        Timer timer("Mini Batch Training Time");

        timer.start();

        nn->miniBatchGradientDescent(2,5000,0.5,false,"");

        timer.stop();

        std::cout<<timer<<std::endl;
    }

    std::cout<<"After Training"<<std::endl;

    std::cout<<"Cost "<<nn->getCost()<<std::endl;

    for(auto i=0;i<input.size();++i) {
        std::vector<long double> actualOutput=nn->predict(input[i]);
        std::cout<<input[i]<<" -> "<<actualOutput<<std::endl;
    }
}
