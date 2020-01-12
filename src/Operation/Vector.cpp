#include "Vector.hpp"

std::vector<long double> Vector::operationOfVector(std::vector<long double>& vector,long double(*function)(long double)) {
    std::vector<long double> output;
    
    for(auto i=0;i<vector.size();++i) {
        output.push_back(function(vector[i]));
    }
    return output;
}

std::vector<long double> Vector::additionOfVector(std::vector<long double>& one,std::vector<long double>& two) {
    std::vector<long double> output;
    
    for(auto i=0;i<one.size();++i) {
        output.push_back(one[i]+two[i]);
    }
    
    return output;
}

std::vector<long double> Vector::subtractionOfVector(std::vector<long double>& one,std::vector<long double>& two) {
    std::vector<long double> output;
    
    for(auto i=0;i<one.size();++i) {
        output.push_back(one[i]-two[i]);
    }
    
    return output;
}


std::vector<long double> Vector::multiplicationOfVector(std::vector<long double>& one,std::vector<long double>& two) {
    std::vector<long double> output;
    
    for(auto i=0;i<one.size();++i) {
        output.push_back(one[i]*two[i]);
    }
    
    return output;
}


std::vector<long double> Vector::divisionOfVector(std::vector<long double>& one,std::vector<long double>& two) {
    std::vector<long double> output;
    
    for(auto i=0;i<one.size();++i) {
        output.push_back(one[i]/two[i]);
    }
    
    return output;
}

std::vector<long double> Vector::multiplyVectorBy(std::vector<long double>& vector,long double value) {
    std::vector<long double> output;
    
    for(auto i=0;i<vector.size();++i) {
        output.push_back(vector[i]*value);
    }
    
    return output;
}

std::vector<long double> Vector::averageOfVector(std::vector<std::vector<long double>>& matrix) {
    std::vector<long double> output;
    
    for(auto i=0;i<matrix[0].size();++i) {
        long double value=0.0;
        
        for(auto x=0;x<matrix.size();++x) {
            value+=matrix[x][i];
        }
        
        value/=(long double)matrix.size();
        
        output.push_back(value);
    }
    
    return output;
}

double long Vector::sumOfVector(std::vector<long double>& vector) {
    long double value=0.0;
    
    for(auto i=0;i<vector.size();++i) {
        value+=vector[i];
    }
    
    return value;
}
