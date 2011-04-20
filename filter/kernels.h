#pragma once

#ifndef CUDA_KERNELS
#define CUDA_KERNELS
#include "Bacalar/cuda/variables.h"


/*
	USEFUL (AND ESTHETICALLY AWFUL) MACRA
*/

#define SE_TO_SHARED(incVar)\
	for(incVar = 0;gpuNbSize[seIndex] > incVar*blockDim.x; incVar++){\
	if(threadIdx.x + blockDim.x*incVar < gpuNbSize[seIndex]){	/*copy only contents of nb array*/\
		nb[threadIdx.x + blockDim.x*incVar] = gpuNb[seIndex][threadIdx.x + blockDim.x*incVar];\
	}}

#define MAP_THREADS_ONTO_IMAGE(incVar)\
	(gpuFrameSize + incVar/gpuImageSliceArea)*gpuImageSliceSize;\
	incVar = incVar % gpuImageSliceArea;\
	arrIdx += (gpuFrameSize + incVar/gpuImageWidth)*gpuImageLineSize\
			+ (gpuFrameSize + incVar%gpuImageWidth)

/*
	KERNELS
*/

template<typename imDataType>
__global__ void GPUerode(imDataType* dst, int seIndex, imDataType* srcA){

			//copy nbhood into shared memory
	extern __shared__ unsigned nb[];	//points to per block allocated shared memory
	unsigned thread = 0;				//temporary usage as incremental varible

			//will run multiple times only if nbsize > number of threads
	SE_TO_SHARED(thread);
	__syncthreads();

			//compute actual index to image array
	thread = threadIdx.x + blockIdx.x*blockDim.x;	//proper usage as global thread ID
	
	if(thread >= gpuImageSize) return;				//terminate excessive threads
	unsigned arrIdx = MAP_THREADS_ONTO_IMAGE(thread);

			//erode (find min)
	imDataType _min = tex1Dfetch(uchar1DTextRef,arrIdx + nb[0]);
	for(thread = 1; thread < gpuNbSize[seIndex]; thread++){
		if(_min > tex1Dfetch(uchar1DTextRef,arrIdx + nb[thread])){
			_min = tex1Dfetch(uchar1DTextRef,arrIdx + nb[thread]);
		}
	}
	dst[arrIdx] = _min;
}

/*
	Using shared memory as a cache
*/
//template<typename imDataType>
//__global__ void GPUerode(imDataType* dst, int seIndex, imDataType* srcA){
//
//			//copy nbhood into shared memory
//	extern __shared__ unsigned nb[];	//points to per block allocated shared memory
//	unsigned thread = 0;				//temporary usage as incremental varible
//
//			//will run multiple times only if nbsize > number of threads
//	SE_TO_SHARED(thread);
//	__syncthreads();
//
//			//compute actual index to image array
//	thread = threadIdx.x + blockIdx.x*blockDim.x;	//proper usage as global thread ID
//	
//	if(thread >= gpuImageSize) return;				//terminate excessive threads
//	unsigned arrIdx = MAP_THREADS_ONTO_IMAGE(thread);
//
//			//erode (find min)
//	imDataType _min = tex1Dfetch(uchar1DTextRef,arrIdx + nb[0]);
//	for(thread = 1; thread < gpuNbSize[seIndex]; thread++){
//		if(_min > tex1Dfetch(uchar1DTextRef,arrIdx + nb[thread])){
//			_min = tex1Dfetch(uchar1DTextRef,arrIdx + nb[thread]);
//		}
//	}
//	dst[arrIdx] = _min;
//}

/*
	Currently only unsigned char
*/
template<typename imDataType>
__global__ void GPUmedian(imDataType* dst, int seIndex, imDataType* srcA){

			//copy nbhood into shared memory
	extern __shared__ unsigned char dynArray[];
	unsigned *nb = (unsigned*)dynArray;	//points to per block allocated shared memory
	unsigned thread = 0;						//temporary usage as incremental varible

			//will run multiple times only if nbsize > number of threads
	SE_TO_SHARED(thread);
	__syncthreads();

			//attach array for partial sorting
	unsigned char *sortArr = dynArray + gpuNbSize[seIndex]*sizeof(unsigned) 
							+ (gpuNbSize[seIndex]/2 + 2)*threadIdx.x;

			//compute actual index to image array
	thread = threadIdx.x + blockIdx.x*blockDim.x;	//proper usage as global thread ID
	
	if(thread >= gpuImageSize) return;				//terminate excessive threads
	unsigned arrIdx = MAP_THREADS_ONTO_IMAGE(thread);

			//fill/initialize array for partial sorting
	imDataType *_min, *_max;
	for(thread = 0;thread<(gpuNbSize[seIndex]/2 + 2); thread++){
		
	}
}

#endif