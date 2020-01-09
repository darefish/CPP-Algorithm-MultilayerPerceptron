#ifndef File_hpp
#define File_hpp

#include <vector>

#include <fstream>

class File {
public:
    static std::vector<std::vector<long double>> readAsMatrix(const char* directory,unsigned long long int length);
    
    static std::vector<long double> readAsVector(const char* directory);
};

#endif
