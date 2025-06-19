#include "cuFFT.h"

#include <iostream>

cuFFT::cuFFT(unsigned int size) : N(size) {
    int rank = 1;
    int n[] = {N};
    int inembed_row[] = {N, N};
    int istride_row = 1, idist_row = N;
    cufftPlanMany(&plan_row, rank, n,
                    inembed_row, istride_row, idist_row,
                    inembed_row, istride_row, idist_row,
                    CUFFT_C2C, N);

    int inembed_col[] = {N, N};
    int istride_col = N, idist_col = 1;
    cufftPlanMany(&plan_col, rank, n,
                    inembed_col, istride_col, idist_col,
                    inembed_col, istride_col, idist_col,
                    CUFFT_C2C, N);

    cudaMalloc(&d_buffer, 2 * N * N * sizeof(cufftComplex));
}

cuFFT::~cuFFT() {
    cufftDestroy(plan_row);
    cufftDestroy(plan_col);
    cudaFree(d_buffer);
}

void cuFFT::batch_fft(complex* h_data, bool is_row) {
    cudaMemcpy(d_buffer, h_data, 2*N*N*sizeof(cufftComplex), cudaMemcpyHostToDevice);

    if(is_row) {
        cufftExecC2C(plan_row, d_buffer, d_buffer, CUFFT_FORWARD);
    } else {
        cufftExecC2C(plan_col, d_buffer, d_buffer, CUFFT_FORWARD);
    }

    cudaMemcpy(h_data, d_buffer, 2*N*N*sizeof(cufftComplex), cudaMemcpyDeviceToHost);
}