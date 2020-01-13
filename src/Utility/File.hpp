#ifndef File_hpp
#define File_hpp

#include <vector>

#include <fstream>

class File {
public:
    static std::vector<std::vector<long double>> readAsMatrix(const char* directory,unsigned long long int interval);
    
    static std::vector<long double> readAsVector(const char* directory);
    
    static void writeAsVector(const char* directory,std::vector<long double>& vector);
};

#endif
