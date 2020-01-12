#include "ActivationFunction.hpp"
 
long double ActivationFunction::sigmoid(long double value) {
    return 1/(1+exp(-value));
}

long double ActivationFunction::derivativeSigmoid(long double value) {
    return value*(1-value);
}

long double ActivationFunction::hyperbolicTangent(long double value) {
    return tanh(value);
}

long double ActivationFunction::derivativeHyperbolicTangent(long double value) {
    return 1-pow(value,2);
}
