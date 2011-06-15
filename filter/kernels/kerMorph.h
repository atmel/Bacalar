#pragma once

#ifndef CUDA_KERNELS_MORPH
#define CUDA_KERNELS_MORPH
#include "Bacalar/cuda/variables.h"


/*
	USEFUL (AND ESTHETICALLY AWFUL) MACRA
*/

#define SE_TO_SHARED(incVar)\
	for(incVar = 0;gpuCap[seIndex] > incVar*blockDim.x; incVar++){\
	if(threadIdx.x + blockDim.x*incVar < gpuCap[seIndex]){	/*copy only contents of wList array*/\
		wList[threadIdx.x + blockDim.x*incVar] = gpuWeightedList[seIndex][threadIdx.x + blockDim.x*incVar];\
	}}

#define MAP_THREADS_ONTO_IMAGE(incVar)\
	(gpuFrameSize + incVar/gpuImageSliceArea)*gpuImageSliceSize;\
	incVar = incVar % gpuImageSliceArea;\
	arrIdx += (gpuFrameSize + incVar/gpuImageWidth)*gpuImageLineSize\
			+ (gpuFrameSize + incVar%gpuImageWidth)

/*-------------- KERNELS ---------------------*/

template<typename imDataType>
__global__ void GPUerode(imDataType* dst, int seIndex, imDataType* srcA){

			//copy nbhood into shared memory
	extern __shared__ unsigned wList[];	//points to per block allocated shared memory
	unsigned thread = 0;				//temporary usage as incremental varible

			//will run multiple times only if nbsize > number of threads
	SE_TO_SHARED(thread);
	__syncthreads();

			//compute actual index to image array
	thread = threadIdx.x + blockIdx.x*blockDim.x;	//proper usage as global thread ID
	
	if(thread >= gpuImageSize) return;				//terminate excessive threads
	unsigned arrIdx = MAP_THREADS_ONTO_IMAGE(thread);

			//erode (find min)
	imDataType _min = tex1Dfetch(uchar1DTextRef,arrIdx + wList[0]);
	for(thread = 1; thread < gpuCap[seIndex]; thread++){
		if(_min > tex1Dfetch(uchar1DTextRef,arrIdx + wList[thread])){
			_min = tex1Dfetch(uchar1DTextRef,arrIdx + wList[thread]);
		}
	}
	dst[arrIdx] = _min;
}

/*#####################################################################################*/

template<typename imDataType>
__global__ void GPUedge(imDataType* dst, int seIndex, imDataType* srcA){

			//copy nbhood into shared memory
	extern __shared__ unsigned wList[];	//points to per block allocated shared memory
	unsigned thread = 0;				//temporary usage as incremental varible

			//will run multiple times only if nbsize > number of threads
	SE_TO_SHARED(thread);
	__syncthreads();

			//compute actual index to image array
	thread = threadIdx.x + blockIdx.x*blockDim.x;	//proper usage as global thread ID
	
	if(thread >= gpuImageSize) return;				//terminate excessive threads
	unsigned arrIdx = MAP_THREADS_ONTO_IMAGE(thread);

			//erode (find min)
	imDataType tmp = tex1Dfetch(uchar1DTextRef,arrIdx + wList[0]);
	imDataType _min = tmp, _max = tmp;
	for(thread = 1; thread < gpuCap[seIndex]; thread++){
		tmp = tex1Dfetch(uchar1DTextRef,arrIdx + wList[thread]);
		if(_min > tmp) _min = tmp;
		if(_max < tmp) _max = tmp;
	}
	dst[arrIdx] = _max - _min;
}


/*-------------- FILE END --------------------*/

#undef SE_TO_SHARED
#undef MAP_THREADS_ONTO_IMAGE

#endif
