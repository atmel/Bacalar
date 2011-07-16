#pragma once

#ifndef CUDA_KERNELS_SORT
#define CUDA_KERNELS_SORT
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



/*#####################################################################################*/

/*
	Currently only unsigned char
*/
#define SORT_ARR_SIZE (gpuCap[seIndex]/2+2)
#define INT32_ALIGNED_SORT_ARR_SIZE ((SORT_ARR_SIZE*sizeof(imDataType))/4+1)

template<typename imDataType>
__global__ void GPUmedian(imDataType* dst, int seIndex, imDataType* srcA){

			//copy nbhood into shared memory
	extern __shared__ unsigned wList[];
	unsigned thread = 0;							//temporary usage as incremental varible

			//will run multiple times only if nbsize > number of threads
	SE_TO_SHARED(thread);
	__syncthreads();

			//compute actual index to image array
	thread = threadIdx.x + blockIdx.x*blockDim.x;	//proper usage as global thread ID	
	if(thread >= gpuImageSize) return;				//terminate excessive threads	

	unsigned arrIdx = MAP_THREADS_ONTO_IMAGE(thread);

		//attach array for partial sorting
	imDataType *sortArr = 
		(imDataType*)&wList[gpuCap[seIndex]+INT32_ALIGNED_SORT_ARR_SIZE*threadIdx.x];

		//fill/initialize array for partial sorting, find min, max consequently
	sortArr[0] = tex1Dfetch(uchar1DTextRef,arrIdx + wList[0]);
	imDataType *_min = sortArr, *_max = sortArr;					//init min,max (indexes to sortArr)

	for(thread = 1;thread<SORT_ARR_SIZE; thread++){
		sortArr[thread] = tex1Dfetch(uchar1DTextRef,arrIdx + wList[thread]);
		if(sortArr[thread] > *_max) _max = &(sortArr[thread]);
		if(sortArr[thread] < *_min) _min = &(sortArr[thread]);
	}
			//forgetful sort (loop begins with knowing min, max position)
			//mins are deleted from the [0], maxs from [last] 
	for(int i = 0;;){
			//accomplishing both conditions yelding less div branches (max on min's position...)
		*_min = (_max == &(sortArr[0]))?sortArr[SORT_ARR_SIZE-1-i]:sortArr[0];
		*_max = (_min == &(sortArr[SORT_ARR_SIZE-1-i]))?sortArr[0]:sortArr[SORT_ARR_SIZE-1-i];

			//end?
		if(gpuCap[seIndex]%2){			//to spare one/two elements respectively
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
		sortArr[0] = tex1Dfetch(uchar1DTextRef,arrIdx + wList[SORT_ARR_SIZE+i]);
			
			//find new min and max
		_min = sortArr; _max = sortArr;
		i++;
		for(thread = 1;thread<SORT_ARR_SIZE-i; thread++){
			if(sortArr[thread] > *_max) _max = &(sortArr[thread]);
			if(sortArr[thread] < *_min) _min = &(sortArr[thread]);
		}
	}
}

/*#####################################################################################*/

#undef SORT_ARR_SIZE
#undef INT32_ALIGNED_SORT_ARR_SIZE

#define BES_SORT_ARR_SIZE (((3*gpuCap[seIndex])>>2)+2)	
#define BES_INT32_ALIGNED_SORT_ARR_SIZE (((BES_SORT_ARR_SIZE*sizeof(imDataType))>>2)+1)

/*
	
	BES = ceil(n/4) + floor((n+1)/2) + ceil((n+1)/2) + floor((3n+4)/4)

	At the begining act as forgerful (shared still suffices) and
	find 1,3-quartil, than continue as forgetful, but do not load
	additional value from memory

*/
template<typename imDataType>
__global__ void GPUBES(imDataType* dst, int seIndex, imDataType* srcA){

			//copy nbhood into shared memory
	extern __shared__ unsigned wList[];
	int thread = 0;							//temporary usage as incremental varible
			//will run multiple times only if nbsize > number of threads
	SE_TO_SHARED(thread);
	__syncthreads();
			//compute actual index to image array
	thread = threadIdx.x + blockIdx.x*blockDim.x;	//proper usage as global thread ID	
	if(thread >= gpuImageSize) return;				//terminate excessive threads	

	unsigned arrIdx = MAP_THREADS_ONTO_IMAGE(thread);
		//attach array for partial sorting
	imDataType *sortArr = 
		(imDataType*)&wList[gpuCap[seIndex]+BES_INT32_ALIGNED_SORT_ARR_SIZE*threadIdx.x];
		//fill/initialize array for partial sorting, find min, max consequently
	sortArr[0] = tex1Dfetch(uchar1DTextRef,arrIdx + wList[0]);

	imDataType *_min = sortArr, *_max = sortArr;					//init min,max (indexes to sortArr)
	for(thread = 1;thread<BES_SORT_ARR_SIZE; thread++){
		sortArr[thread] = tex1Dfetch(uchar1DTextRef,arrIdx + wList[thread]);
		if(sortArr[thread] > *_max) _max = &(sortArr[thread]);
		if(sortArr[thread] < *_min) _min = &(sortArr[thread]);
	}
			//forgetful sort (loop begins with knowing min, max position)
			//mins are deleted from the [0], maxs from [last] 
	int i;
	for(i = 0;;){
			//accomplishing both conditions yelding less div branches (max on min's position...)
		*_min = (_max == &(sortArr[0]))?sortArr[BES_SORT_ARR_SIZE-1-i]:sortArr[0];
		*_max = (_min == &(sortArr[BES_SORT_ARR_SIZE-1-i]))?sortArr[0]:sortArr[BES_SORT_ARR_SIZE-1-i];

			//end? (no unsorted element to include?)
		if(BES_SORT_ARR_SIZE+i == gpuCap[seIndex]) break;

			//move new unsorted to [0] -- array shrinks from top
		sortArr[0] = tex1Dfetch(uchar1DTextRef,arrIdx + wList[BES_SORT_ARR_SIZE+i]);
			
			//find new min and max
		_min = sortArr; _max = sortArr;
		i++;
		for(thread = 1;thread<BES_SORT_ARR_SIZE-i; thread++){
			if(sortArr[thread] > *_max) _max = &(sortArr[thread]);
			if(sortArr[thread] < *_min) _min = &(sortArr[thread]);
		}
	}
		//now, next min,max will be the 1,4-quartil
	i+=2;
	sortArr[0] = sortArr[BES_SORT_ARR_SIZE-i];	//so algorithm can be the same
	_min = sortArr; _max = sortArr;
	for(thread = 1;thread<BES_SORT_ARR_SIZE-i; thread++){
		if(sortArr[thread] > *_max) _max = &(sortArr[thread]);
		if(sortArr[thread] < *_min) _min = &(sortArr[thread]);
	}
		//store 1,4-quartil
	sortArr[BES_SORT_ARR_SIZE-1] = *_max;
	sortArr[BES_SORT_ARR_SIZE-2] = *_min;
	//unsigned partSum = (*_max) + (*_min);

	while(1){
			//accomplishing both conditions yelding less div branches (max on min's position...)
		*_min = (_max == &(sortArr[0]))?sortArr[BES_SORT_ARR_SIZE-1-i]:sortArr[0];
		*_max = (_min == &(sortArr[BES_SORT_ARR_SIZE-1-i]))?sortArr[0]:sortArr[BES_SORT_ARR_SIZE-1-i];

			//end?
		if(gpuCap[seIndex]%2){			//to spare one/two elements respectively
			if(BES_SORT_ARR_SIZE-i <= 3){		//odd	
				dst[arrIdx] = (unsigned)((unsigned)sortArr[BES_SORT_ARR_SIZE-1] +
					sortArr[BES_SORT_ARR_SIZE-2] + (unsigned)2*sortArr[1])/4;	//position of the median
				return;
			}
		}else{
			if(BES_SORT_ARR_SIZE-i <= 4){		//even	
				dst[arrIdx] = (unsigned)((unsigned)sortArr[BES_SORT_ARR_SIZE-1] +
					sortArr[BES_SORT_ARR_SIZE-2] + (unsigned)sortArr[1] + sortArr[2])/4;
				return;
			}
		}
			//move new unsorted to [0] -- array shrinks from top
		i+=2;
		sortArr[0] = sortArr[BES_SORT_ARR_SIZE-i];
			
			//find new min and max
		_min = sortArr; _max = sortArr;
		for(thread = 1;thread<BES_SORT_ARR_SIZE-i; thread++){
			if(sortArr[thread] > *_max) _max = &(sortArr[thread]);
			if(sortArr[thread] < *_min) _min = &(sortArr[thread]);
		}
	}
}

/*#####################################################################################*/

#undef SE_TO_SHARED
#undef MAP_THREADS_ONTO_IMAGE

#endif