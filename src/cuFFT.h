#ifndef CUFFT_H
#define CUFFT_H

#include <math.h>
#include "cuda_complex.h"
#include "complex.h"
#include <cuda_runtime.h>
#include <cufft.h>

class cuFFT {
private:
    int N;
    cufftHandle plan_row;
    cufftHandle plan_col;
    cufftComplex* d_buffer;

public:
    cuFFT(unsigned int N);
    ~cuFFT();
    void batch_fft(complex* h_data, bool is_row);
};

#endif