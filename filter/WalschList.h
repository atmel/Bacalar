#pragma once

#ifndef WALSH_LIST
#define WALSH_LIST

#include <Bacalar/Filter.h>


#define HODGES_THREAD_PER_BLOCK (128)
	//+8 = +2 types for lower and higher median, +2 control
#define HODGES_ADD_MEM_SIZE_PER_BLOCK \
	(((sem->GetSE(seIndex)->nbSize*(sem->GetSE(seIndex)->nbSize+1)+8)/2)*sizeof(imDataType))

template <typename imDataType>
float Filter<imDataType>::WMedian(imDataType* dst, int seIndex, imDataType* srcA, fourthParam<imDataType> p4){
	LARGE_INTEGER frq, start, stop;
	QueryPerformanceFrequency(&frq);

	if(UseCuda()){
		//cout << "median on GPU\n";
		unsigned blocks = GetImageSize()/HODGES_THREAD_PER_BLOCK + 1;
		//cout << blocks <<" kernel blocks used\n";
		unsigned extraMem = (sem->GetSE(seIndex)->nbSize)*sizeof(unsigned)
				+ HODGES_ADD_MEM_SIZE_PER_BLOCK;
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

		GPUhodgesmed<imDataType><<<blocks,HODGES_THREAD_PER_BLOCK,extraMem>>>(dst,seIndex,srcA);
		cudaThreadSynchronize();
			QueryPerformanceCounter(&stop);

		return (double)((stop.QuadPart-start.QuadPart)*1000)/frq.QuadPart;

	}else{
		return 1;
	}
}

#endif