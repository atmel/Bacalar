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
#define SORT_ARR_SIZE (gpuNbSize[seIndex]/2+2)

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
							+ SORT_ARR_SIZE*threadIdx.x;

			//compute actual index to image array
	thread = threadIdx.x + blockIdx.x*blockDim.x;	//proper usage as global thread ID
	
	if(thread >= gpuImageSize) return;				//terminate excessive threads
	unsigned arrIdx = MAP_THREADS_ONTO_IMAGE(thread);

			//fill/initialize array for partial sorting, find min, max consequently
	sortArr[0] = tex1Dfetch(uchar1DTextRef,arrIdx + nb[0]);
	imDataType *_min = sortArr, *_max = sortArr, tmp;	//init min,max

	for(thread = 1;thread<SORT_ARR_SIZE; thread++){
		sortArr[thread] = tex1Dfetch(uchar1DTextRef,arrIdx + nb[thread]);
		if(sortArr[thread] > *_max) _max = &sortArr[thread];
		if(sortArr[thread] < *_min) _min = &sortArr[thread];
	}
			//forgetful sort (loop begins with knowing min, max position)
			//mins are deleted from the [0], maxs from [last] 
	for(int i = 0;;){
			//forget min
		tmp = sortArr[SORT_ARR_SIZE-1-i];	//in case _min points to the end
		*_min = sortArr[0];
			//forget max
		*_max = tmp;

			//end?
		if(gpuNbSize[seIndex]%2){			//to spare one/two elements respectively
			if(SORT_ARR_SIZE-i <= 3){		//odd	
				dst[arrIdx] = sortArr[1];	//position of the median
				return;
			}
		}else{
			if(SORT_ARR_SIZE-i <= 4){		//even	
				dst[arrIdx] = ((unsigned)sortArr[1]+sortArr[2])/2;
				return;
			}
		}
			//move new unsorted to [0] -- array shrinks from top
		sortArr[0] = tex1Dfetch(uchar1DTextRef,arrIdx + nb[SORT_ARR_SIZE-1+i]);
			
			//find new min and max
		_min = sortArr; _max = sortArr;
		i++;
		for(thread = 1;thread<SORT_ARR_SIZE-i; thread++){
			if(sortArr[thread] > *_max) _max = &sortArr[thread];
			if(sortArr[thread] < *_min) _min = &sortArr[thread];
		}
	}
}

#endif