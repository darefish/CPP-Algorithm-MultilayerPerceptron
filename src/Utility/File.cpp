#include "File.hpp"

std::vector<std::vector<long double>> File::readAsMatrix(const char* directory,unsigned long long int interval) {
    
    std::vector<std::vector<long double>> output;
    
    std::fstream file(directory);
    
    if(file.is_open()) {
        unsigned long long int count=0;
        
        long double value=0.0;
        
        std::vector<long double> local;
        
        while(file>>value) {
            file.ignore();
            if(count%interval==0 && count!=0) {
                output.push_back(local);
                local.clear();
            }
            local.push_back(value);
            
            count++;
        }
        file.close();
        
        output.push_back(local);
        
    }
    
    return output;
}

std::vector<long double> File::readAsVector(const char* directory) {
    std::vector<long double> output;
    
    std::fstream file(directory);
    
    if(file.is_open()) {
        long double value=0.0;
        
        while(file>>value) {
            output.push_back(value);
        }
        
        file.close();
    }
    
    return output;
}

void File::writeAsVector(const char* directory,std::vector<long double>& vector) {
    std::ofstream file(directory);
    
    for(auto i=0;i<vector.size();++i) {
        file<<vector[i]<<"\n";
    }
    
    file.close();
}
