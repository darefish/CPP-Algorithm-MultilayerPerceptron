#ifndef ActivationFunction_hpp
#define ActivationFunction_hpp

#include <cmath>

class ActivationFunction {
public:
    
    static long double sigmoid(long double value);
    
    static long double derivativeSigmoid(long double value);
    
    static long double hyperbolicTangent(long double value);
    
    static long double derivativeHyperbolicTangent(long double value);
    
    static long double RELU(long double value);
    
    static long double derivativeRELU(long double value);
};

#endif
