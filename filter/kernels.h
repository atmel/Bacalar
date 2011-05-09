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

/*#####################################################################################*/

template<typename imDataType>
__global__ void GPUedge(imDataType* dst, int seIndex, imDataType* srcA){

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
	imDataType tmp = tex1Dfetch(uchar1DTextRef,arrIdx + nb[0]);
	imDataType _min = tmp, _max = tmp;
	for(thread = 1; thread < gpuNbSize[seIndex]; thread++){
		tmp = tex1Dfetch(uchar1DTextRef,arrIdx + nb[thread]);
		if(_min > tmp) _min = tmp;
		if(_max < tmp) _max = tmp;
	}
	dst[arrIdx] = _max - _min;
}

/*#####################################################################################*/

/*
	Currently only unsigned char
*/
#define SORT_ARR_SIZE ((gpuNbSize[seIndex]/2+2)*sizeof(imDataType))
#define INT32_ALIGNED_SORT_ARR_SIZE (SORT_ARR_SIZE/4+1)

template<typename imDataType>
__global__ void GPUmedian(imDataType* dst, int seIndex, imDataType* srcA){

			//copy nbhood into shared memory
	extern __shared__ unsigned nb[];
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
		(imDataType*)&nb[gpuNbSize[seIndex]+INT32_ALIGNED_SORT_ARR_SIZE*threadIdx.x];

		//fill/initialize array for partial sorting, find min, max consequently
	sortArr[0] = tex1Dfetch(uchar1DTextRef,arrIdx + nb[0]);
	imDataType *_min = sortArr, *_max = sortArr;					//init min,max (indexes to sortArr)

	for(thread = 1;thread<SORT_ARR_SIZE; thread++){
		sortArr[thread] = tex1Dfetch(uchar1DTextRef,arrIdx + nb[thread]);
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
		sortArr[0] = tex1Dfetch(uchar1DTextRef,arrIdx + nb[SORT_ARR_SIZE+i]);
			
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

#define BES_SORT_ARR_SIZE (((3*gpuNbSize[seIndex])/4+2)*sizeof(imDataType))
#define BES_INT32_ALIGNED_SORT_ARR_SIZE (BES_SORT_ARR_SIZE/4+1)

/*
	
	BES = ceil(n/4) + floor((n+1)/2) + ceil((n+1)/2) + floor((3n+4)/4)

	At the begining act as forgerful (shared still suffices) and
	find 1,4-quartil, than continue as forgetful, but do not load
	additional value from memory

*/
template<typename imDataType>
__global__ void GPUBES(imDataType* dst, int seIndex, imDataType* srcA){

			//copy nbhood into shared memory
	extern __shared__ unsigned nb[];
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
		(imDataType*)&nb[gpuNbSize[seIndex]+BES_INT32_ALIGNED_SORT_ARR_SIZE*threadIdx.x];
		//fill/initialize array for partial sorting, find min, max consequently
	sortArr[0] = tex1Dfetch(uchar1DTextRef,arrIdx + nb[0]);

	imDataType *_min = sortArr, *_max = sortArr;					//init min,max (indexes to sortArr)
	for(thread = 1;thread<BES_SORT_ARR_SIZE; thread++){
		sortArr[thread] = tex1Dfetch(uchar1DTextRef,arrIdx + nb[thread]);
		if(sortArr[thread] > *_max) _max = &(sortArr[thread]);
		if(sortArr[thread] < *_min) _min = &(sortArr[thread]);
	}
			//forgetful sort (loop begins with knowing min, max position)
			//mins are deleted from the [0], maxs from [last] 
	unsigned i;
	for(i = 0;;){
			//accomplishing both conditions yelding less div branches (max on min's position...)
		*_min = (_max == &(sortArr[0]))?sortArr[BES_SORT_ARR_SIZE-1-i]:sortArr[0];
		*_max = (_min == &(sortArr[BES_SORT_ARR_SIZE-1-i]))?sortArr[0]:sortArr[BES_SORT_ARR_SIZE-1-i];

			//end? (no unsorted element to include?)
		if(BES_SORT_ARR_SIZE+i == gpuNbSize[seIndex]) break;

			//move new unsorted to [0] -- array shrinks from top
		sortArr[0] = tex1Dfetch(uchar1DTextRef,arrIdx + nb[BES_SORT_ARR_SIZE+i]);
			
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
		if(gpuNbSize[seIndex]%2){			//to spare one/two elements respectively
			if(BES_SORT_ARR_SIZE-i <= 3){		//odd	
				dst[arrIdx] = ((unsigned)sortArr[BES_SORT_ARR_SIZE-1] +
					sortArr[BES_SORT_ARR_SIZE-2] + 2*sortArr[1])/4;	//position of the median
				return;
			}
		}else{
			if(BES_SORT_ARR_SIZE-i <= 4){		//even	
				dst[arrIdx] = ((unsigned)sortArr[BES_SORT_ARR_SIZE-1] +
					sortArr[BES_SORT_ARR_SIZE-2] + sortArr[1] + sortArr[2])/4;
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


#define HODGES_SORT_ARR_SIZE (((gpuNbSize[seIndex]*(gpuNbSize[seIndex]+1))/2)*sizeof(imDataType))
#define WALSH_NB_TO_SHARED(incVar)\
	for(incVar = 0;gpuNbSize[seIndex] > incVar*blockDim.x; incVar++){\
	if(threadIdx.x + blockDim.x*incVar < gpuNbSize[seIndex]){/*copy only contents of nb array*/\
		sortArr[threadIdx.x + blockDim.x*incVar]=\
		tex1Dfetch(uchar1DTextRef,arrIdx + nb[threadIdx.x + blockDim.x*incVar]);\
	}}
#define ODD (HODGES_SORT_ARR_SIZE%2)

template<typename imDataType>
__global__ void GPUhodgesmed(imDataType* dst, int seIndex, imDataType* srcA){

			//copy nbhood into shared memory
	extern __shared__ unsigned nb[];
	unsigned thread = 0;							//temporary usage as incremental varible
			//will run multiple times only if nbsize > number of threads
	SE_TO_SHARED(thread);
	__syncthreads();

	imDataType *sortArr = (imDataType*)&nb[gpuNbSize[seIndex]];
	unsigned arrIdx, lesser, bigger;

		//process sequentially as much pixels as is block size
	for(unsigned i=0;i<blockDim.x;i++){
		thread = i + blockIdx.x*blockDim.x;
		if(thread >= gpuImageSize) return;			//end (entire block)
		arrIdx = MAP_THREADS_ONTO_IMAGE(thread);
		WALSH_NB_TO_SHARED(thread);
		__syncthreads();
			//generate Walsh list
		for(thread = 0; HODGES_SORT_ARR_SIZE-gpuNbSize[seIndex] > thread*blockDim.x; thread++){
				//determine from which two elemets average is to be calcualted
			unsigned j;
			j = threadIdx.x + blockDim.x*thread;
			if(j >= HODGES_SORT_ARR_SIZE-gpuNbSize[seIndex]) break;		//too high
			for(arrIdx=0;;arrIdx++){							//arrIdx as temporary
				if(j < gpuNbSize[seIndex]-1-arrIdx) break;
				j = j+1+arrIdx-gpuNbSize[seIndex];
			}
				//store averages behind nbpixel values
			sortArr[threadIdx.x+blockDim.x*thread + gpuNbSize[seIndex]] = 
				((unsigned)sortArr[arrIdx]+sortArr[j+arrIdx+1])/2;
		}
		__syncthreads();
			//find median
		sortArr[HODGES_SORT_ARR_SIZE+2] = sortArr[HODGES_SORT_ARR_SIZE+3] = 0;	//control variables
		for(thread = 0; HODGES_SORT_ARR_SIZE > thread*blockDim.x; thread++){	//if array is longer than blockDim
				//terminate excessive threads
			if(threadIdx.x + blockDim.x*thread >= HODGES_SORT_ARR_SIZE) break; 
			if(sortArr[HODGES_SORT_ARR_SIZE+2]&&sortArr[HODGES_SORT_ARR_SIZE+2]) break;	//both medians found, end
			imDataType curElement = sortArr[threadIdx.x + blockDim.x*thread];

				//compare
			bigger = lesser = 0;
			for(arrIdx=0;arrIdx<HODGES_SORT_ARR_SIZE;arrIdx++){	//arrIdx as temporary
				if(sortArr[arrIdx] > curElement) bigger++;
				if(sortArr[arrIdx] < curElement) lesser++;
			}
			if((lesser<=HODGES_SORT_ARR_SIZE/2-1+ODD)&&(bigger<=HODGES_SORT_ARR_SIZE/2)){
				sortArr[HODGES_SORT_ARR_SIZE  ] = curElement;
				sortArr[HODGES_SORT_ARR_SIZE+2] = 1;
			}
			if((lesser<=HODGES_SORT_ARR_SIZE/2)&&(bigger<=HODGES_SORT_ARR_SIZE/2-1+ODD)){ 
				sortArr[HODGES_SORT_ARR_SIZE+1] = curElement;
				sortArr[HODGES_SORT_ARR_SIZE+3] = 1;
			}
		}
		__syncthreads();
		if(threadIdx.x == 0){
			thread = i + blockIdx.x*blockDim.x;
			arrIdx = MAP_THREADS_ONTO_IMAGE(thread);
			dst[arrIdx] = ((unsigned)sortArr[HODGES_SORT_ARR_SIZE]+sortArr[HODGES_SORT_ARR_SIZE+1])/2;
		}
		__syncthreads();
	}
}

#endif