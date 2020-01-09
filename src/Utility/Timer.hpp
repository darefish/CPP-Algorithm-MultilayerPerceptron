#ifndef Timer_hpp
#define Timer_hpp

#include <chrono>

class Timer {
public:
    const char* process;
    
    std::chrono::time_point<std::chrono::steady_clock> begin,end;
    
    std::chrono::duration<long double> duration;
    
    Timer();
    
    Timer(const char* process);
    
    void start();
    
    void stop();
    
    ~Timer();
};

#endif
