#include <cufft.h>
#include <iostream>
 
int main() {
    cufftHandle plan;
    cufftPlan1d(&plan, 1024, CUFFT_Z2Z, 1); // 创建一个1D FFT计划
    cufftDestroy(plan); // 销毁计划
    std::cout << "cuFFT initialized successfully." << std::endl;
    return 0;
}