#include <stdlib.h>
#include <stdio.h>
#include <iostream>
 
#include <string.h>
#include <math.h>
 
#include <cuda_runtime.h>
#include <cufft.h>

#include "src/fft.h"
#include "src/complex.h"
#define Ndim 2
#define NX 8
#define NY 4
 
 
void testplanmany() {
 
	int N[2];
	N[0] = NX, N[1] = NY;
	int NXY = N[0] * N[1];
    complex in[NX*NY];

	cufftComplex *input = (cufftComplex*) malloc(NXY * sizeof(cufftComplex));
	cufftComplex *output = (cufftComplex*) malloc(NXY * sizeof(cufftComplex));
	int i;
	for (i = 0; i < NXY; i++) {
		in[i].a = input[i].x = i % 1000;
		in[i].b = input[i].y = 0;
	}
	cufftComplex *d_inputData, *d_outData;
	cudaMalloc((void**) &d_inputData, N[0] * N[1] * sizeof(cufftComplex));
	cudaMalloc((void**) &d_outData, N[0] * N[1] * sizeof(cufftComplex));
	cudaMemcpy(d_inputData, input, N[0] * N[1] * sizeof(cufftComplex), cudaMemcpyHostToDevice);
    std::cerr <<"check "<< sizeof(d_inputData) <<std::endl;
	cufftHandle plan;
	/*
	cufftMakePlanMany(cufftHandle plan, int rank, int *n, int *inembed,
	int istride, int idist, int *onembed, int ostride,
	int odist, cufftType type, int batch, size_t *workSize);
	 */
	int rank=1;
	int n[1];
	n[0]=NX;
	int istride=1;
	int idist = NX;
	int ostride=1;
	int odist = NX;
	int inembed[2];
	int onembed[2];
	inembed[0]=NX;  onembed[0]=NX;
	inembed[1] = NY; onembed[0] = NY;
 
	cufftPlanMany(&plan,rank,n,inembed, istride ,idist , onembed, ostride,odist, CUFFT_C2C, NY);
	cufftExecC2C(plan, d_inputData, d_outData, CUFFT_FORWARD);
	cudaMemcpy(output, d_outData, NXY * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
 
	for (i = 0; i < NXY; i++) {
		if(i%NX==0)
			std::cerr << std::endl;
		std::cerr << output[i].x << " " << output[i].y << std::endl;
	}

    // for (int i = 0; i < NXY; i++) std::cout <<in[i].a <<' '<< in[i].b<<std::endl;

    cFFT fft(NX);
    for (i = 0; i < NY; i++) {
			fft.fft(in, in, 1, i * NX);
    }

    for (i = 0; i < NXY; i++) {
		if(i%NX==0)
			std::cerr << std::endl;
		std::cerr << in[i].a << " " << in[i].b << std::endl;
	}
 
	cufftDestroy(plan);
	free(input);
	free(output);
	cudaFree(d_inputData);
	cudaFree(d_outData);
}
 
int main() {
 
	testplanmany();
}