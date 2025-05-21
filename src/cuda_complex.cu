#include "cuda_complex.h"

cufftComplex complex_to_cufft(const complex& c) {
    return make_cuComplex(c.a, c.b);
}

complex cufft_to_complex(const cufftComplex& c) {
    return complex(c.x, c.y);
}