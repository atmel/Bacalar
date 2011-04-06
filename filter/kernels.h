#pragma once

#ifndef CUDA_KERNELS
#define CUDA_KERNELS
#include "Bacalar/cuda/variables.cu"

template<typename imDataType>
__global__ void GPUerode(imDataType* dst, int seIndex, imDataType* srcA){

		//copy nbhood into shared memory
	extern __shared__ unsigned nb[];	//points to per block allocated shared memory
	unsigned thread = 0;				//temporary usage as incremental varible

	for(thread = 0;gpuNbSize[seIndex] > thread*blockDim.x; thread++){
		if(threadIdx.x + blockDim.x*thread < gpuNbSize[seIndex]){	//copy only contents of nb array
			nb[threadIdx.x + blockDim.x*thread] = gpuNb[seIndex][threadIdx.x + blockDim.x*thread];
		}
	}
	__syncthreads();

		//compute actual index to image array
	thread = threadIdx.x + blockIdx.x*blockDim.x;	//proper usage as global thread ID
	if(thread >= gpuImageSize) return;				//terminate excessive threads
	unsigned arrIdx = (gpuFrameSize + thread/gpuImageSliceArea)*gpuImageSliceSize;
	thread = thread % gpuImageSliceArea;
	arrIdx += (gpuFrameSize + thread/gpuImageWidth)*gpuImageLineSize + (gpuFrameSize + thread%gpuImageWidth);

		//erode (find min)
	imDataType _min = srcA[arrIdx + nb[0]];
	for(thread = 1; thread < gpuNbSize[seIndex]; thread++){
		if(_min > srcA[arrIdx + nb[thread]]){
			_min = srcA[arrIdx + nb[thread]];
		}
	}
	dst[arrIdx] = _min;
}


#endif