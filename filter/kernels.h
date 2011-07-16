//#pragma once
//
//#ifndef CUDA_KERNELS
//#define CUDA_KERNELS
//#include "Bacalar/cuda/variables.h"
//
//
///*
//	USEFUL (AND ESTHETICALLY AWFUL) MACRA
//*/
//
//#define SE_TO_SHARED(incVar)\
//	for(incVar = 0;gpuCap[seIndex] > incVar*blockDim.x; incVar++){\
//	if(threadIdx.x + blockDim.x*incVar < gpuCap[seIndex]){	/*copy only contents of wList array*/\
//		wList[threadIdx.x + blockDim.x*incVar] = gpuWeightedList[seIndex][threadIdx.x + blockDim.x*incVar];\
//	}}
//
//#define MAP_THREADS_ONTO_IMAGE(incVar)\
//	(gpuFrameSize + incVar/gpuImageSliceArea)*gpuImageSliceSize;\
//	incVar = incVar % gpuImageSliceArea;\
//	arrIdx += (gpuFrameSize + incVar/gpuImageWidth)*gpuImageLineSize\
//			+ (gpuFrameSize + incVar%gpuImageWidth)
//
///*
//	KERNELS
//*/
//
//
//
//#endif