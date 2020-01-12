#include "Matrix.hpp"

std::vector<std::vector<long double>> Matrix::isolationOfMatrix(std::vector<std::vector<std::vector<long double>>>& vector,unsigned long long int layer) {
    std::vector<std::vector<long double>> output;
    
    for(unsigned long long int i=0;i<vector.size();++i) {
        output.push_back(vector[i][layer]);
    }
    
    return output;
}

std::vector<std::vector<long double>> Matrix::averageOfMatrix(std::vector<std::vector<std::vector<long double>>>& vector) {
    
    std::vector<std::vector<long double>> output;
    
    for(unsigned long long int i=0;i<vector[0].size();++i) {
        
        std::vector<std::vector<long double>> local=isolationOfMatrix(vector,i);
        
        output.push_back(Vector::averageOfVector(local));
    }
    
    return output;
}
