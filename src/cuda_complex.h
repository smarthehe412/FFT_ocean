#ifndef CUDA_COMPLEX_H
#define CUDA_COMPLEX_H

#include <cufft.h>
#include "complex.h"

cufftComplex complex_to_cufft(const complex& c);
complex cufft_to_complex(const cufftComplex& c);

#endif