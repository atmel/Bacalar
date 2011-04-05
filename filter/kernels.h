#pragma once

#ifndef CUDA_KERNELS
#define CUDA_KERNELS
#include "Bacalar/cuda/variables.cu"

template<typename imDataType>
__global__ void GPUerode(imDataType* dst, int seIndex, imDataType* srcA){

		//compute actual index to image array
	unsigned thread = threadIdx.x + blockIdx.x*blockDim.x;
	if(thread >= gpuImageSize) return;		//terminate excessive threads
	unsigned arrIdx = (gpuFrameSize + thread/gpuImageSliceArea)*gpuImageSliceSize;
	thread = thread % gpuImageSliceArea;
	arrIdx += (gpuFrameSize + thread/gpuImageWidth)*gpuImageLineSize + (gpuFrameSize + thread%gpuImageWidth);

		//erode (find min)
	imDataType _min = srcA[arrIdx + gpuNb[seIndex][0]];
	for(thread = 1; thread < gpuNbSize[seIndex]; thread++){
		if(_min > srcA[arrIdx + gpuNb[seIndex][thread]]){
			_min = srcA[arrIdx + gpuNb[seIndex][thread]];
		}
	}
	dst[arrIdx] = _min;
}


#endif