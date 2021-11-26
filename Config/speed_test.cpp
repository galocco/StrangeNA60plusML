#include "TStopwatch.h"
#include "TVector.h"
void speed_test(){
    TStopwatch watch;
    watch.Stat();
    TVector v1;
    for(int i =0; i<100000;i++){
        v1 = TVector(i,i,i);
    }
    std::cout<<"watch 1: "<<watch.CpuTime()<<std::endl;

    watch.Stat();
    for(int i =0; i<100000;i++){
        TVector v2(i,i,i);
    }
    std::cout<<"watch 2: "<<watch.CpuTime()<<std::endl;
    
    watch.Stat();
    for(int i =0; i<100000;i++){
        TVector* v3 = new TVector(i,i,i);
    }
    std::cout<<"watch 3: "<<watch.CpuTime()<<std::endl;
    
}