#pragma once

#ifndef MORPHOLOGY
#define MORPHOLOGY

/*

	Contains morphology filters (not using mask values - treating them as 1/0)

*/

#include "Bacalar/Filter.h"
#include <windows.h>

#define TPB (128)

template <typename imDataType>
float Filter<imDataType>::Erode(imDataType* dst, int seIndex, imDataType* srcA, fourthParam<imDataType> p4){
	LARGE_INTEGER frq, start, stop;
	QueryPerformanceFrequency(&frq);
	if(UseCuda()){
		cout << "eroding on GPU\n";
		unsigned blocks = GetImageSize()/TPB + 1;
		cout << blocks <<" kernel blocks used\n";
		unsigned extraMem = (sem->GetSE(seIndex)->nbSize)*sizeof(unsigned);
		cout << extraMem <<" extra mem per block used\n";
			//bind texture
		uchar1DTextRef.normalized = false;
		uchar1DTextRef.addressMode[0] = cudaAddressModeClamp;
		uchar1DTextRef.filterMode = cudaFilterModePoint;

		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<imDataType>();

		cudaBindTexture(0,&uchar1DTextRef,(void*)srcA,&channelDesc,GetTotalPixelSize()*sizeof(imDataType));
		cout << "binding texture cuda error:" << cudaGetErrorString(cudaGetLastError()) << '\n';
		
			QueryPerformanceCounter(&start);

		GPUerode<imDataType><<<blocks,TPB,extraMem>>>(dst,seIndex,srcA);
		cudaThreadSynchronize();
			QueryPerformanceCounter(&stop);

			cout << "GpuErode ticks: " << 
				(double)((stop.QuadPart-start.QuadPart)*1000)/frq.QuadPart << "miliseconds\n";
		
		cout << "erode cuda error:" << cudaGetErrorString(cudaGetLastError()) << '\n';
		return 1;
	}else{
		imDataType min;
		structEl *se = sem->GetSE(seIndex);
		memset(dst,0,GetTotalPixelSize()*sizeof(imDataType));
		unsigned long pos;

			QueryPerformanceCounter(&start);
		//3D
		BEGIN_FOR3D(pos)			
			min = srcA[pos + se->nb[0]];		//find minimum
			for(unsigned m=1;m < se->nbSize; m++){
				if(min > srcA[pos + se->nb[m]])
					min = srcA[pos + se->nb[m]];
			}
			dst[pos] = min;
		END_FOR3D;

			QueryPerformanceCounter(&stop);

			cout << "CPUErode ticks: " << 
				(double)((stop.QuadPart-start.QuadPart)*1000)/frq.QuadPart << "miliseconds\n";
		return 1;
	}
}


template <typename imDataType>
float Filter<imDataType>::Dilatate(imDataType* dst, int seIndex, imDataType* srcA, fourthParam<imDataType> p4){
	imDataType max;
	structEl *se = sem->GetSE(seIndex);
	memset(dst,0,GetTotalPixelSize()*sizeof(imDataType));
	unsigned long pos;

	//3D
	BEGIN_FOR3D(pos)			
		max = srcA[pos + se->nb[0]];		//find minimum
		for(unsigned m=1;m < se->nbSize; m++){
			if(max < srcA[pos + se->nb[m]])
				max = srcA[pos + se->nb[m]];
		}
		dst[pos] = max;
	END_FOR3D;
	return 1;
}


template <typename imDataType>
float Filter<imDataType>::Open (imDataType* dst, int seIndex, imDataType* srcA, fourthParam<imDataType> p4){
	return 1; 
}
template <typename imDataType>
float Filter<imDataType>::Close (imDataType* dst, int seIndex, imDataType* srcA, fourthParam<imDataType> p4){
	return 1; 
}

#endif
