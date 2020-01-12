#ifndef Matrix_hpp
#define Matrix_hpp

#include <vector>

#include "Vector.hpp"

class Matrix {
public:
    static std::vector<std::vector<long double>> isolationOfMatrix(std::vector<std::vector<std::vector<long double>>>& vector,unsigned long long int layer);
    
    static std::vector<std::vector<long double>> averageOfMatrix(std::vector<std::vector<std::vector<long double>>>& vector);
};

#endif
