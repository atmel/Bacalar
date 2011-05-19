#pragma once

#ifndef WALSH_LIST
#define WALSH_LIST

#include <Bacalar/Filter.h>


#define HODGES_THREAD_PER_BLOCK (64)
#define HODGES_OPT_THREADS_PER_PIXEL (16)
	//+8 = +2 types for lower and higher median, +2 control
#define HODGES_ADD_MEM_SIZE_PER_BLOCK \
	(((((sem->GetSE(seIndex)->nbSize*(sem->GetSE(seIndex)->nbSize+1)+8)/2)*sizeof(imDataType))/4+1)*sizeof(unsigned))
#define HODGES_CPU_SORT_ARR_SIZE \
	((sem->GetSE(seIndex)->nbSize*(sem->GetSE(seIndex)->nbSize+1))/2)
#define HODGES_FORGET_ADD_MEM_PER_BLOCK \
	((((((sem->GetSE(seIndex)->nbSize*(sem->GetSE(seIndex)->nbSize+1))/4+2)*sizeof(imDataType)))/4+1)*sizeof(unsigned))
#define HODGES_OPT_WALSH_INDEXES_SIZE \
	((((sem->GetSE(seIndex)->nbSize*(sem->GetSE(seIndex)->nbSize-1))*sizeof(unsigned char))/4+1)*sizeof(unsigned))

#define HODGES_FIRST_TOTAL_ADD_MEM HODGES_ADD_MEM_SIZE_PER_BLOCK
#define HODGES_OPT_TOTAL_ADD_MEM \
	(HODGES_ADD_MEM_SIZE_PER_BLOCK*(HODGES_THREAD_PER_BLOCK/HODGES_OPT_THREADS_PER_PIXEL)\
	+HODGES_OPT_WALSH_INDEXES_SIZE)
#define HODGES_FORGET_TOTAL_ADD_MEM (HODGES_FORGET_ADD_MEM_PER_BLOCK*HODGES_THREAD_PER_BLOCK)

template <typename imDataType>
float Filter<imDataType>::WMedian(imDataType* dst, int seIndex, imDataType* srcA, fourthParam<imDataType> p4){
	LARGE_INTEGER frq, start, stop;
	QueryPerformanceFrequency(&frq);

	if(UseCuda()){
		//cout << "median on GPU\n";
		unsigned blocks = GetImageSize()/HODGES_THREAD_PER_BLOCK + 1;
		//cout << blocks <<" kernel blocks used\n";
		unsigned extraMem = (sem->GetSE(seIndex)->nbSize)*sizeof(unsigned)
				+ HODGES_OPT_TOTAL_ADD_MEM;
		cout << extraMem <<" extra mem per block used\n";
		//cout << "ADD_MEM_SIZE_PER_BLOCK: " << BES_ADD_MEM_SIZE_PER_BLOCK <<'\n';
		//cout << "arr size: " << ((sem->GetSE(seIndex)->nbSize)/2+2) <<'\n';
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
		imDataType *values;
		structEl *se = sem->GetSE(seIndex);
		memset(dst,0,GetTotalPixelSize()*sizeof(imDataType));
		values = new imDataType[HODGES_CPU_SORT_ARR_SIZE];
		Filter<imDataType>::MedianFindOpt(NULL,HODGES_CPU_SORT_ARR_SIZE);		//initialize median
		unsigned pos;

		//3D
			QueryPerformanceCounter(&start);

		unsigned m,n,o;
		BEGIN_FOR3D(pos)	
			for(m=0;m < se->nbSize; m++){
				values[m] = srcA[pos + se->nb[m]];
			}	
			o = se->nbSize;
			for(m=0; m < se->nbSize-1; m++){
				for(n=m+1; n < se->nbSize; n++,o++)
					values[o] = ((unsigned)values[m]+values[n])/2;
			}	

			Filter<imDataType>::MedianFindOpt(values);
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

#endif