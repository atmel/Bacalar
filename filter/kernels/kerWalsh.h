#pragma once

#ifndef CUDA_KERNELS_WALSH
#define CUDA_KERNELS_WALSH
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

/*------------------ KERNELS ---------------------*/

#define HODGES_SORT_ARR_SIZE ((gpuCap[seIndex]*(gpuCap[seIndex]+1))/2)
#define WALSH_NB_TO_SHARED(incVar)\
	for(incVar = 0;gpuCap[seIndex] > incVar*blockDim.x; incVar++){\
	if(threadIdx.x + blockDim.x*incVar < gpuCap[seIndex]){/*copy only contents of wList array*/\
		sortArr[threadIdx.x + blockDim.x*incVar]=\
		tex1Dfetch(uchar1DTextRef,arrIdx + wList[threadIdx.x + blockDim.x*incVar]);\
	}}
#define ODD (HODGES_SORT_ARR_SIZE%2)

	//there are two indexes for each arithmetic average
#define INT32_ALIGNED_WALSH_INDEXES_SIZE \
	(((gpuCap[seIndex]*(gpuCap[seIndex]-1))*sizeof(unsigned char))/4+1)
	//spare a register? At least not force a register
//#define WALSH_IDX_ARRAY(idx) (unsigned char*)&wList[gpuCap[seIndex]]+idx
#define GENERATE_WALSH_INDEXES(incVar,incVarInner,tmpVar) \
	for(incVar = 0; HODGES_SORT_ARR_SIZE-gpuCap[seIndex] > incVar*blockDim.x; incVar++){\
		tmpVar = threadIdx.x + blockDim.x*incVar;\
		if(tmpVar >= HODGES_SORT_ARR_SIZE-gpuCap[seIndex]) break;\
		for(incVarInner=1; tmpVar>=gpuCap[seIndex]-incVarInner; incVarInner++){\
			tmpVar-= -incVarInner+gpuCap[seIndex];\
		}\
		walshIdxArr[2*(threadIdx.x + blockDim.x*incVar)  ] = incVarInner-1;\
		walshIdxArr[2*(threadIdx.x + blockDim.x*incVar)+1] = incVarInner+tmpVar;\
	}

#define THREADS_PER_PIXEL (16)
	// +4 for control variables
#define INT32_ALIGNED_HODGES_SORT_ARR_SIZE \
	(((HODGES_SORT_ARR_SIZE+4)*sizeof(imDataType))/4+1)
	//array index is approprietally set for each subgroup
#define WALSH_NB_TO_SHARED_PARALELL(incVar,tmpVar)\
	for(incVar = 0,tmpVar = threadIdx.x%THREADS_PER_PIXEL;\
	 gpuCap[seIndex] > incVar;\
	 incVar += THREADS_PER_PIXEL, tmpVar += THREADS_PER_PIXEL){\
			/*do not write behind sortArr*/\
		if(tmpVar >= gpuCap[seIndex]) break;\
		sortArr[tmpVar]=tex1Dfetch(uchar1DTextRef,arrIdx + wList[tmpVar]);\
	}

/*

	Process more pixels per one run (to hide overhead of creating walsch list indexes, copying SE to shared and so on)
		pixels with index 0-THREADS_PER_PIXEL will process 1st pixel and so on. Block will
		process a total of (blockDim/THREADS_PER_PIXEL)*blockDim pixels, which helps to hide latencies
		as memory request can be covered by computation

	Synchronization: if group of threads should be mapped onto pixel out of image, it does nothing but synchonize
		instead. If it is the case, value of 'thread' never changes and no new variable is needed
TODO> check if thread can exceed the 


*/
template<typename imDataType>
__global__ void GPUhodgesmedOpt(imDataType* dst, int seIndex, imDataType* srcA){

			//copy nbhood into shared memory
	extern __shared__ unsigned wList[];
	unsigned thread = 0;							//temporary usage as incremental varible
			//will run multiple times only if nbsize > number of threads
	SE_TO_SHARED(thread);
	__syncthreads();

		//for storing indexes to generate wlist (would it be faster to precompute and load from glob. mem?)
	unsigned char *walshIdxArr = (unsigned char*)&wList[gpuCap[seIndex]];
		//each warp (generally group of threads) has its own sortArr
	imDataType *sortArr = (imDataType*)&wList[gpuCap[seIndex] + INT32_ALIGNED_WALSH_INDEXES_SIZE 
			+ INT32_ALIGNED_HODGES_SORT_ARR_SIZE*(threadIdx.x/THREADS_PER_PIXEL)];
	unsigned arrIdx, lesser, bigger;

		//precompute wlist indexes
	GENERATE_WALSH_INDEXES(arrIdx, lesser, bigger)
	__syncthreads();

		//process sequentially as much pixels as is block size
		//more than one pixel is processed at a time
	for(unsigned i=0;i<blockDim.x; i += blockDim.x/THREADS_PER_PIXEL){	//jump over few pixels
		thread = i + blockIdx.x*blockDim.x + threadIdx.x/THREADS_PER_PIXEL;
		if(thread < gpuImageSize){						//WHAT THE SHIT?!?
			arrIdx = MAP_THREADS_ONTO_IMAGE(thread);
			WALSH_NB_TO_SHARED_PARALELL(thread, bigger);
				//arrIdx not used until storage of the result -> use it as dummy variable
		}
		__syncthreads();
			//generate Walsh list
		//if(thread < gpuImageSize){
		//for(thread = 0; HODGES_SORT_ARR_SIZE-gpuCap[seIndex] > thread*THREADS_PER_PIXEL; thread++){
		//		//store averages behind nbpixel values
		//	if((threadIdx.x%THREADS_PER_PIXEL)+THREADS_PER_PIXEL*thread >= HODGES_SORT_ARR_SIZE-gpuCap[seIndex]) goto SYNC1;
		//	sortArr[(threadIdx.x % THREADS_PER_PIXEL) + THREADS_PER_PIXEL*thread + gpuCap[seIndex]] =
		//		((unsigned)sortArr[walshIdxArr[2*((threadIdx.x%THREADS_PER_PIXEL) + THREADS_PER_PIXEL*thread)+1]]
		//		+sortArr[walshIdxArr[2*((threadIdx.x%THREADS_PER_PIXEL) + THREADS_PER_PIXEL*thread)]])/2;
		//}
		//}
		if(thread < gpuImageSize){
			for(thread = 0, arrIdx = threadIdx.x%THREADS_PER_PIXEL; 
			 HODGES_SORT_ARR_SIZE-gpuCap[seIndex] > thread; 
			 thread += THREADS_PER_PIXEL, arrIdx += THREADS_PER_PIXEL){
					//store averages behind nbpixel values
				if(arrIdx >= HODGES_SORT_ARR_SIZE-gpuCap[seIndex]) goto SYNC1;
				sortArr[arrIdx + gpuCap[seIndex]] =
					((unsigned)sortArr[walshIdxArr[2*(arrIdx)+1]]
					+sortArr[walshIdxArr[2*(arrIdx)]])/2;
			}
			thread = 0; //to ensure that condition works
		}
SYNC1:
		__syncthreads();
			//find median
			//use arridx as dummy instead, thread is reserved for overflow handling
		sortArr[HODGES_SORT_ARR_SIZE+2] = sortArr[HODGES_SORT_ARR_SIZE+3] = 0;	//control variables
		for(arrIdx = 0; HODGES_SORT_ARR_SIZE > arrIdx; arrIdx+=THREADS_PER_PIXEL){	//if array is longer than blockDim
			if(thread < gpuImageSize){		//if 
					//terminate excessive threads
				thread = (threadIdx.x%THREADS_PER_PIXEL) + arrIdx;
				if(thread >= HODGES_SORT_ARR_SIZE) goto SYNC2; 
				if(sortArr[HODGES_SORT_ARR_SIZE+2]&&sortArr[HODGES_SORT_ARR_SIZE+2]) goto SYNC2; //both medians found, end
				imDataType curElement = sortArr[thread];

					//compare
				bigger = lesser = 0;
				for(thread=0;thread<HODGES_SORT_ARR_SIZE;thread++){	//thread as temporary
					bigger += (sortArr[thread] > curElement);		//faster, no branching
					lesser += (sortArr[thread] < curElement);
				}
				if(!((lesser>HODGES_SORT_ARR_SIZE/2)||(bigger>HODGES_SORT_ARR_SIZE/2))){
					if((lesser<=HODGES_SORT_ARR_SIZE/2-1+ODD)&&(bigger<=HODGES_SORT_ARR_SIZE/2)){
						sortArr[HODGES_SORT_ARR_SIZE  ] = curElement;
						sortArr[HODGES_SORT_ARR_SIZE+2] = 1;
					}
					if((lesser<=HODGES_SORT_ARR_SIZE/2)&&(bigger<=HODGES_SORT_ARR_SIZE/2-1+ODD)){ 
						sortArr[HODGES_SORT_ARR_SIZE+1] = curElement;
						sortArr[HODGES_SORT_ARR_SIZE+3] = 1;
					}
				}
				thread = 0;	//to ensure that condition works
			}
SYNC2:
			__syncthreads();
		}
		if(thread < gpuImageSize){
			if((threadIdx.x%THREADS_PER_PIXEL) == 0){
				thread = i + blockIdx.x*blockDim.x + threadIdx.x/THREADS_PER_PIXEL;
				arrIdx = MAP_THREADS_ONTO_IMAGE(thread);
				dst[arrIdx] = ((unsigned)sortArr[HODGES_SORT_ARR_SIZE]+sortArr[HODGES_SORT_ARR_SIZE+1])/2;
			}
			thread = 0; //to ensure that condition works
		}
		__syncthreads();
	}
}

/*####################################################################################################*/

template<typename imDataType>
__global__ void GPUhodgesmedfirst(imDataType* dst, int seIndex, imDataType* srcA){

			//copy nbhood into shared memory
	extern __shared__ unsigned wList[];
	unsigned thread = 0;							//temporary usage as incremental varible
			//will run multiple times only if nbsize > number of threads
	SE_TO_SHARED(thread);
	__syncthreads();

	imDataType *sortArr = (imDataType*)&wList[gpuCap[seIndex]];
	unsigned arrIdx, lesser, bigger;

		//process sequentially as much pixels as is block size
	for(unsigned i=0;i<blockDim.x;i++){
		thread = i + blockIdx.x*blockDim.x;
		if(thread >= gpuImageSize) return;		//whole block
		arrIdx = MAP_THREADS_ONTO_IMAGE(thread);
		WALSH_NB_TO_SHARED(thread);
		__syncthreads();
			//generate Walsh list
		for(thread = 0; HODGES_SORT_ARR_SIZE-gpuCap[seIndex] > thread*blockDim.x; thread++){
				//determine from which two elemets average is to be calcualted
			unsigned j;
			j = threadIdx.x + blockDim.x*thread;
			if(j >= HODGES_SORT_ARR_SIZE-gpuCap[seIndex]) goto SYNC1;	//too high
			for(arrIdx=1; j>=gpuCap[seIndex]-arrIdx; arrIdx++){										//arrIdx as temporary
				j-= -arrIdx+gpuCap[seIndex];
			}
				//store averages behind nbpixel values
			sortArr[threadIdx.x+blockDim.x*thread + gpuCap[seIndex]] = 
				((unsigned)sortArr[arrIdx-1]+sortArr[j+arrIdx])/2;
SYNC1:
			__syncthreads();
		}
			//find median
		sortArr[HODGES_SORT_ARR_SIZE+2] = sortArr[HODGES_SORT_ARR_SIZE+3] = 0;	//control variables
		for(thread = 0; HODGES_SORT_ARR_SIZE > thread*blockDim.x; thread++){	//if array is longer than blockDim
				//terminate excessive threads
			if(threadIdx.x + blockDim.x*thread >= HODGES_SORT_ARR_SIZE) goto SYNC2; 
			if(sortArr[HODGES_SORT_ARR_SIZE+2]&&sortArr[HODGES_SORT_ARR_SIZE+2]) goto SYNC2; //both medians found, end
			imDataType curElement = sortArr[threadIdx.x + blockDim.x*thread];

				//compare
			bigger = lesser = 0;
			for(arrIdx=0;arrIdx<HODGES_SORT_ARR_SIZE;arrIdx++){	//arrIdx as temporary
				bigger += (sortArr[arrIdx] > curElement);		//faster, no branching
				lesser += (sortArr[arrIdx] < curElement);
			}
			if(!((lesser>HODGES_SORT_ARR_SIZE/2)||(bigger>HODGES_SORT_ARR_SIZE/2))){
				if((lesser<=HODGES_SORT_ARR_SIZE/2-1+ODD)&&(bigger<=HODGES_SORT_ARR_SIZE/2)){
					sortArr[HODGES_SORT_ARR_SIZE  ] = curElement;
					sortArr[HODGES_SORT_ARR_SIZE+2] = 1;
				}
				if((lesser<=HODGES_SORT_ARR_SIZE/2)&&(bigger<=HODGES_SORT_ARR_SIZE/2-1+ODD)){ 
					sortArr[HODGES_SORT_ARR_SIZE+1] = curElement;
					sortArr[HODGES_SORT_ARR_SIZE+3] = 1;
				}
			}
SYNC2:
			__syncthreads();
		}
		if(threadIdx.x == 0){
			thread = i + blockIdx.x*blockDim.x;
			arrIdx = MAP_THREADS_ONTO_IMAGE(thread);
			dst[arrIdx] = ((unsigned)sortArr[HODGES_SORT_ARR_SIZE]+sortArr[HODGES_SORT_ARR_SIZE+1])/2;
		}
		__syncthreads();
	}
}

/*####################################################################################################*/

#define HODGES2_SORT_ARR_SIZE ((gpuCap[seIndex]*(gpuCap[seIndex]+1))/4+2)
#define HODGES2_INT32_ALIGNED_SORT_ARR_SIZE ((HODGES2_SORT_ARR_SIZE*sizeof(imDataType))/4+1)

template<typename imDataType>
__global__ void GPUhodgesmedforget(imDataType* dst, int seIndex, imDataType* srcA){

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
		(imDataType*)&wList[gpuCap[seIndex]+HODGES2_INT32_ALIGNED_SORT_ARR_SIZE*threadIdx.x];

		//fill/initialize array for partial sorting, find min, max consequently
	sortArr[0] = tex1Dfetch(uchar1DTextRef,arrIdx + wList[0]);
	imDataType *_min = sortArr, *_max = sortArr;					//init min,max (indexes to sortArr)

		//load nbhood
	for(thread = 1;thread<gpuCap[seIndex]; thread++){
		sortArr[thread] = tex1Dfetch(uchar1DTextRef,arrIdx + wList[thread]);
		if(sortArr[thread] > *_max) _max = &(sortArr[thread]);
		if(sortArr[thread] < *_min) _min = &(sortArr[thread]);
	}
		//generate walsh list (backwards)
	int i,j;
	for(i = gpuCap[seIndex]-1; i > 0; i--){
		for(j = gpuCap[seIndex]; j > i; j--){
			if(thread >= HODGES2_SORT_ARR_SIZE) goto FINISH_INIT;	//only fill array
			sortArr[thread] = ((unsigned)sortArr[i]+sortArr[j])/2;
			if(sortArr[thread] > *_max) _max = &(sortArr[thread]);
			if(sortArr[thread] < *_min) _min = &(sortArr[thread]);
			thread++;
		}
	}
FINISH_INIT:
			//forgetful sort (loop begins with knowing min, max position)
			//mins are deleted from the [0], maxs from [last] 
	for(i = 0;;){
			//accomplishing both conditions yelding less div branches (max on min's position...)
		*_min = (_max == &(sortArr[0]))?sortArr[HODGES2_SORT_ARR_SIZE-1-i]:sortArr[0];
		*_max = (_min == &(sortArr[HODGES2_SORT_ARR_SIZE-1-i]))?sortArr[0]:sortArr[HODGES2_SORT_ARR_SIZE-1-i];

			//end?
		if(gpuCap[seIndex]%2){			//to spare one/two elements respectively
			if(HODGES2_SORT_ARR_SIZE-i <= 3){		//odd	
				dst[arrIdx] = sortArr[1];	//position of the median
				return;
			}
		}else{
			if(HODGES2_SORT_ARR_SIZE-i <= 4){		//even	
				dst[arrIdx] = ((unsigned)sortArr[1]+sortArr[2])/2;
				return;
			}
		}
			//move new unsorted to [0] -- array shrinks from top
			//use thread, j as temporary variables
		thread = i;		//will be consumed/changed
			//same process as in GPUhodgesmed
		for(j=1; thread>=gpuCap[seIndex]-j; j++){
			thread-= -j+gpuCap[seIndex];
		}
		sortArr[0] = ((unsigned)sortArr[j-1]+sortArr[j+thread])/2;
			
			//find new min and max
		_min = sortArr; _max = sortArr;
		i++;
		for(thread = 1;thread<HODGES2_SORT_ARR_SIZE-i; thread++){
			if(sortArr[thread] > *_max) _max = &(sortArr[thread]);
			if(sortArr[thread] < *_min) _min = &(sortArr[thread]);
		}
	}
}

/*------------------ FILE END --------------------*/

#undef SE_TO_SHARED
#undef MAP_THREADS_ONTO_IMAGE

#endif