#include "Neuron.hpp"

Neuron::Neuron() {
    
}

void Neuron::connect(Neuron* neuron) {
    connections.push_back(neuron);
    weights.push_back((long double)rand()/(long double)RAND_MAX);
}

void Neuron::activate() {
    for(auto i=0;i<connections.size();++i) {
        connections[i]->value+=this->value*weights[i];
    }
}

Neuron::~Neuron() {
    
}
