#ifndef Neuron_hpp
#define Neuron_hpp

#include <vector>

class Neuron {
public:
    long double value=0.0;
    
    long double bias=0.0;
    
    std::vector<Neuron*> connections;
    
    std::vector<long double> weights;
    
    Neuron();
    
    void connect(Neuron* neuron);
    
    void activate();
    
    ~Neuron();
};

#endif
