#ifndef Vector_hpp
#define Vector_hpp

#include <vector>

class Vector {
public:
    static std::vector<long double> operationOfVector(std::vector<long double>& vector,long double(*function)(long double));
    
    static std::vector<long double> additionOfVector(std::vector<long double>& one,std::vector<long double>& two);
    
    static std::vector<long double> subtractionOfVector(std::vector<long double>& one,std::vector<long double>& two);
    
    static std::vector<long double> multiplicationOfVector(std::vector<long double>& one,std::vector<long double>& two);
    
    static std::vector<long double> divisionOfVector(std::vector<long double>& one,std::vector<long double>& two);
    
    static std::vector<long double> multiplyVectorBy(std::vector<long double>& vector,long double value);
    
    static std::vector<long double> averageOfVector(std::vector<std::vector<long double>>& matrix);
    
    static long double sumOfVector(std::vector<long double>& vector);
};

#endif
