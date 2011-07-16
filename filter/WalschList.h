#pragma once

#ifndef WALSH_LIST
#define WALSH_LIST

#include <Bacalar/Filter.h>


#define HODGES_THREAD_PER_BLOCK (32)
#define HODGES_OPT_THREADS_PER_PIXEL (16)
	//+8 = +2 types for lower and higher median, +2 control
#define HODGES_ADD_MEM_SIZE_PER_BLOCK \
	(((((sem->GetSE(seIndex)->capacity*(sem->GetSE(seIndex)->capacity+1)+8)/2)*sizeof(imDataType))/4+1)*sizeof(unsigned))
#define HODGES_CPU_SORT_ARR_SIZE \
	((sem->GetSE(seIndex)->capacity*(sem->GetSE(seIndex)->capacity+1))/2)
//#define HODGES_FORGET_ADD_MEM_PER_BLOCK \
//	((((((sem->GetSE(seIndex)->capacity*(sem->GetSE(seIndex)->capacity+1))/4+2)*sizeof(imDataType)))/4+1)*sizeof(unsigned))
#define HODGES_OPT_WALSH_INDEXES_SIZE \
	((((sem->GetSE(seIndex)->capacity*(sem->GetSE(seIndex)->capacity-1))*sizeof(unsigned char))/4+1)*sizeof(unsigned))

//#define HODGES_FIRST_TOTAL_ADD_MEM HODGES_ADD_MEM_SIZE_PER_BLOCK
#define HODGES_OPT_TOTAL_ADD_MEM \
	(HODGES_ADD_MEM_SIZE_PER_BLOCK*(HODGES_THREAD_PER_BLOCK/HODGES_OPT_THREADS_PER_PIXEL)\
	+HODGES_OPT_WALSH_INDEXES_SIZE)
//#define HODGES_FORGET_TOTAL_ADD_MEM (HODGES_FORGET_ADD_MEM_PER_BLOCK*HODGES_THREAD_PER_BLOCK)

template <typename imDataType>
float Filter<imDataType>::WMedian(imDataType* dst, int seIndex, imDataType* srcA, fourthParam<imDataType> p4){
	LARGE_INTEGER frq, start, stop;
	QueryPerformanceFrequency(&frq);

	if(UseCuda()){
		//cout << "median on GPU\n";
		unsigned blocks = GetImageSize()/HODGES_THREAD_PER_BLOCK + 1;
		//cout << blocks <<" kernel blocks used\n";
		unsigned extraMem = (sem->GetSE(seIndex)->capacity)*sizeof(unsigned)
				+ HODGES_OPT_TOTAL_ADD_MEM;
		cout << extraMem <<" extra mem per block used\n";
		//cout << "ADD_MEM_SIZE_PER_BLOCK: " << BES_ADD_MEM_SIZE_PER_BLOCK <<'\n';
		//cout << "arr size: " << ((sem->GetSE(seIndex)->capacity)/2+2) <<'\n';
			//bind texture
		uchar1DTextRef.normalized = false;
		uchar1DTextRef.addressMode[0] = cudaAddressModeClamp;
		uchar1DTextRef.filterMode = cudaFilterModePoint;

		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<imDataType>();

		cudaBindTexture(0,&uchar1DTextRef,(void*)srcA,&channelDesc,GetTotalPixelSize()*sizeof(imDataType));
		//cout << "binding texture cuda error:" << cudaGetErrorString(cudaGetLastError()) << '\n';
		
			QueryPerformanceCounter(&start);
	//-------------------
		GPUhodgesmedOpt<imDataType><<<blocks,HODGES_THREAD_PER_BLOCK,extraMem>>>(dst,seIndex,srcA);
	//-------------------
		cudaThreadSynchronize();
			QueryPerformanceCounter(&stop);

		return (float)((stop.QuadPart-start.QuadPart)*1000)/frq.QuadPart;

	}else{
		imDataType *values;//, *values2;
		structEl *se = sem->GetSE(seIndex);
		memset(dst,0,GetTotalPixelSize()*sizeof(imDataType));
		values = new imDataType[HODGES_CPU_SORT_ARR_SIZE];
		//values2 = new imDataType[HODGES_CPU_SORT_ARR_SIZE];
		//Filter<imDataType>::QsortOpt(NULL,HODGES_CPU_SORT_ARR_SIZE);		//initialize median
		Filter<imDataType>::MedianFindOptSimple(NULL,HODGES_CPU_SORT_ARR_SIZE);		//initialize median
		unsigned pos;

		//3D
			QueryPerformanceCounter(&start);

		unsigned m,n,o;
		BEGIN_FOR3D(pos)	
			for(m=0;m < se->capacity; m++){
				values[m] = srcA[pos + se->wList[m]];
				//values2[m] = srcA[pos + se->wList[m]];
			}	
			o = se->capacity;
			for(m=0; m < se->capacity-1; m++){
				for(n=m+1; n < se->capacity; n++,o++){
					values[o] = ((unsigned)values[m]+values[n])/2;
					//values2[o] = ((unsigned)values2[m]+values2[n])/2;
				}
			}	

			//Filter<imDataType>::QsortOpt(values);
			//cout << "bla\n";
			Filter<imDataType>::MedianFindOptSimple(values);
			//cout << "bla\n";
			//for(m=0;m<HODGES_CPU_SORT_ARR_SIZE;m++){
			//	values[m] -= values[m];		//if it works fine
			//}

			if(HODGES_CPU_SORT_ARR_SIZE%2){							//odd
				dst[pos] = values[HODGES_CPU_SORT_ARR_SIZE/2];
			}else{
				dst[pos] = ((unsigned)values[HODGES_CPU_SORT_ARR_SIZE/2-1]+values[HODGES_CPU_SORT_ARR_SIZE/2])/2;
			}
		END_FOR3D;

			QueryPerformanceCounter(&stop);

		return (float)((stop.QuadPart-start.QuadPart)*1000)/frq.QuadPart;
	}
}

/*#########################################################################################################*/

#define WBES_THREAD_PER_BLOCK (64)
#define WBES_OPT_THREADS_PER_PIXEL (32)
	//+16 = +2 types for lower and higher median, +2 quarils, +4 control
#define WBES_ADD_MEM_SIZE_PER_BLOCK \
	(((((sem->GetSE(seIndex)->capacity*(sem->GetSE(seIndex)->capacity+1)+16)/2)*sizeof(imDataType))/4+1)*sizeof(unsigned))
#define WBES_CPU_SORT_ARR_SIZE \
	((sem->GetSE(seIndex)->capacity*(sem->GetSE(seIndex)->capacity+1))/2)
//#define WBES_FORGET_ADD_MEM_PER_BLOCK \
//	((((((sem->GetSE(seIndex)->capacity*(sem->GetSE(seIndex)->capacity+1))/4+2)*sizeof(imDataType)))/4+1)*sizeof(unsigned))
#define WBES_OPT_WALSH_INDEXES_SIZE \
	((((sem->GetSE(seIndex)->capacity*(sem->GetSE(seIndex)->capacity-1))*sizeof(unsigned char))/4+1)*sizeof(unsigned))

//#define WBES_FIRST_TOTAL_ADD_MEM WBES_ADD_MEM_SIZE_PER_BLOCK
#define WBES_OPT_TOTAL_ADD_MEM \
	(WBES_ADD_MEM_SIZE_PER_BLOCK*(WBES_THREAD_PER_BLOCK/WBES_OPT_THREADS_PER_PIXEL)\
	+WBES_OPT_WALSH_INDEXES_SIZE)
//#define WBES_FORGET_TOTAL_ADD_MEM (WBES_FORGET_ADD_MEM_PER_BLOCK*WBES_THREAD_PER_BLOCK)

template <typename imDataType>
float Filter<imDataType>::WBES(imDataType* dst, int seIndex, imDataType* srcA, fourthParam<imDataType> p4){
	LARGE_INTEGER frq, start, stop;
	QueryPerformanceFrequency(&frq);

	if(UseCuda()){
		//cout << "median on GPU\n";
		unsigned blocks = GetImageSize()/WBES_THREAD_PER_BLOCK + 1;
		//cout << blocks <<" kernel blocks used\n";
		unsigned extraMem = (sem->GetSE(seIndex)->capacity)*sizeof(unsigned)
				+ WBES_OPT_TOTAL_ADD_MEM;
		//cout << extraMem <<" extra mem per block used\n";
		//cout << "ADD_MEM_SIZE_PER_BLOCK: " << BES_ADD_MEM_SIZE_PER_BLOCK <<'\n';
		//cout << "arr size: " << ((sem->GetSE(seIndex)->capacity)/2+2) <<'\n';
			//bind texture
		uchar1DTextRef.normalized = false;
		uchar1DTextRef.addressMode[0] = cudaAddressModeClamp;
		uchar1DTextRef.filterMode = cudaFilterModePoint;

		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<imDataType>();

		cudaBindTexture(0,&uchar1DTextRef,(void*)srcA,&channelDesc,GetTotalPixelSize()*sizeof(imDataType));
		//cout << "binding texture cuda error:" << cudaGetErrorString(cudaGetLastError()) << '\n';
		
			QueryPerformanceCounter(&start);
	//-------------------
		GPUwbesOpt<imDataType><<<blocks,WBES_THREAD_PER_BLOCK,extraMem>>>(dst,seIndex,srcA);
	//-------------------
		cudaThreadSynchronize();
			QueryPerformanceCounter(&stop);

		return (float)((stop.QuadPart-start.QuadPart)*1000)/frq.QuadPart;

	}else{
		imDataType *values;//, *values2;
		structEl *se = sem->GetSE(seIndex);
		memset(dst,0,GetTotalPixelSize()*sizeof(imDataType));
		values = new imDataType[WBES_CPU_SORT_ARR_SIZE];
		//values2 = new imDataType[WBES_CPU_SORT_ARR_SIZE];
		//Filter<imDataType>::QsortOpt(NULL,WBES_CPU_SORT_ARR_SIZE);	
		Filter<imDataType>::UniBESFind(NULL,WBES_CPU_SORT_ARR_SIZE);		//initialize WBESfind
		unsigned pos;
		unsigned q1pos = ceil((float)WBES_CPU_SORT_ARR_SIZE/4)-1;				//BES constatants
		unsigned q3pos = floor((float)(3*WBES_CPU_SORT_ARR_SIZE+4)/4)-1;

		//3D
			QueryPerformanceCounter(&start);

		unsigned m,n,o;
		BEGIN_FOR3D(pos)	
			for(m=0;m < se->capacity; m++){
				values[m] = srcA[pos + se->wList[m]];
				//values2[m] = srcA[pos + se->wList[m]];
			}	
			o = se->capacity;
			for(m=0; m < se->capacity-1; m++){
				for(n=m+1; n < se->capacity; n++,o++){
					values[o] = ((unsigned)values[m]+values[n])/2;
					//values2[o] = ((unsigned)values2[m]+values2[n])/2;
				}
			}	

			//Filter<imDataType>::QsortOpt(values);
			//cout << "bla\n";
			Filter<imDataType>::UniBESFind(values);
			//cout << "bla\n";
			//for(m=0;m<WBES_CPU_SORT_ARR_SIZE;m++){
			//	values[m] -= values[m];		//if it works fine
			//}

			if(WBES_CPU_SORT_ARR_SIZE%2){							//odd
				dst[pos] = ((unsigned)values[q1pos]+2*values[WBES_CPU_SORT_ARR_SIZE/2]+
					values[q3pos])/4;
			}else{
				dst[pos] = ((unsigned)values[WBES_CPU_SORT_ARR_SIZE/2-1]+values[WBES_CPU_SORT_ARR_SIZE/2]+
					values[q1pos]+values[q3pos])/4;
			}
		END_FOR3D;

			QueryPerformanceCounter(&stop);

		return (float)((stop.QuadPart-start.QuadPart)*1000)/frq.QuadPart;
	}
}


#endif