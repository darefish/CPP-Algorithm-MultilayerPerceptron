#include "Timer.hpp"

Timer::Timer() {
    process="Process";
}

Timer::Timer(const char* process) {
    this->process=process;
}

void Timer::start() {
    begin=std::chrono::steady_clock::now();
}

void Timer::stop() {
    end=std::chrono::steady_clock::now();
    
    duration=end-begin;
}

Timer::~Timer() {
    
}
