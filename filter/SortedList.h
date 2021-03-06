#pragma once

#ifndef SORTED_LIST
#define SORTED_LIST

#include "Bacalar/Filter.h"

#define THREAD_PER_BLOCK_MED (192)		//13 registers best occupancy estimate
#define ADD_MEM_SIZE_PER_BLOCK (((((sem->GetSE(seIndex)->capacity)/2+2)*sizeof(imDataType))/4+1)*sizeof(unsigned))

template <typename imDataType>
float Filter<imDataType>::Median(imDataType* dst, int seIndex, imDataType* srcA, fourthParam<imDataType> p4){
	LARGE_INTEGER frq, start, stop;
	QueryPerformanceFrequency(&frq);


	if(UseCuda()){
		//cout << "median on GPU\n";
		unsigned blocks = GetImageSize()/THREAD_PER_BLOCK_MED + 1;
		//cout << blocks <<" kernel blocks used\n";
		unsigned extraMem = (sem->GetSE(seIndex)->capacity)*sizeof(unsigned)
				+ THREAD_PER_BLOCK_MED*ADD_MEM_SIZE_PER_BLOCK;
		/*cout << extraMem <<" extra mem per block used\n";
		cout << "ADD_MEM_SIZE_PER_BLOCK: " << ADD_MEM_SIZE_PER_BLOCK <<'\n';
		cout << "arr size: " << ((sem->GetSE(seIndex)->capacity)/2+2) <<'\n';*/
			//bind texture
		uchar1DTextRef.normalized = false;
		uchar1DTextRef.addressMode[0] = cudaAddressModeClamp;
		uchar1DTextRef.filterMode = cudaFilterModePoint;

		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<imDataType>();

		cudaBindTexture(0,&uchar1DTextRef,(void*)srcA,&channelDesc,GetTotalPixelSize()*sizeof(imDataType));
		//cout << "binding texture cuda error:" << cudaGetErrorString(cudaGetLastError()) << '\n';
		
			QueryPerformanceCounter(&start);

		GPUmedian<imDataType><<<blocks,THREAD_PER_BLOCK_MED,extraMem>>>(dst,seIndex,srcA);
		cudaThreadSynchronize();
			QueryPerformanceCounter(&stop);

		return (double)((stop.QuadPart-start.QuadPart)*1000)/frq.QuadPart;

	}else{
		imDataType *values;
		structEl *se = sem->GetSE(seIndex);
		memset(dst,0,GetTotalPixelSize()*sizeof(imDataType));
		values = new imDataType[se->capacity];
		Filter<imDataType>::MedianFindOpt(NULL,se->capacity);		//initialize median
		unsigned long pos;

		//3D
			QueryPerformanceCounter(&start);

		BEGIN_FOR3D(pos)	
			for(unsigned m=0;m < se->capacity; m++){
				values[m] = srcA[pos + se->wList[m]];
			}		

			Filter<imDataType>::MedianFindOpt(values);
			if(se->capacity%2){							//odd
				dst[pos] = values[se->capacity/2];
			}else{
				dst[pos] = ((unsigned)values[se->capacity/2-1]+values[se->capacity/2])/2;
			}
		END_FOR3D;

			QueryPerformanceCounter(&stop);

		return (double)((stop.QuadPart-start.QuadPart)*1000)/frq.QuadPart;
	}
}

#define BES_THREAD_PER_BLOCK (192)
#define BES_ADD_MEM_SIZE_PER_BLOCK (((((3*sem->GetSE(seIndex)->capacity)/4+2)*sizeof(imDataType))/4+1)*sizeof(unsigned))

template <typename imDataType>
float Filter<imDataType>::BES(imDataType* dst, int seIndex, imDataType* srcA, fourthParam<imDataType> p4){
	LARGE_INTEGER frq, start, stop;
	QueryPerformanceFrequency(&frq);

	if(UseCuda()){
		//cout << "median on GPU\n";
		unsigned blocks = GetImageSize()/BES_THREAD_PER_BLOCK + 1;
		//cout << blocks <<" kernel blocks used\n";
		unsigned extraMem = (sem->GetSE(seIndex)->capacity)*sizeof(unsigned)
				+ BES_THREAD_PER_BLOCK*BES_ADD_MEM_SIZE_PER_BLOCK;
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

		GPUBES<imDataType><<<blocks,BES_THREAD_PER_BLOCK,extraMem>>>(dst,seIndex,srcA);
		cudaThreadSynchronize();
			QueryPerformanceCounter(&stop);

		return (double)((stop.QuadPart-start.QuadPart)*1000)/frq.QuadPart;

	}else{
		imDataType *values;//, *values2;
		structEl *se = sem->GetSE(seIndex);
		memset(dst,0,GetTotalPixelSize()*sizeof(imDataType));
		values = new imDataType[se->capacity];
		//values2 = new imDataType[se->capacity];
		//Filter<imDataType>::QsortOpt(NULL,se->capacity);
		Filter<imDataType>::UniBESFind(NULL,se->capacity);		//initialize median
		unsigned long pos;
		unsigned q1 = ceil((float)se->capacity/4)-1;
		unsigned q3 = floor((float)(3*se->capacity+4)/4)-1;

		//3D
			QueryPerformanceCounter(&start);

		BEGIN_FOR3D(pos)	
			for(unsigned m=0;m < se->capacity; m++){
				values[m] = srcA[pos + se->wList[m]];
				//values2[m] = srcA[pos + se->wList[m]];
			}		

			Filter<imDataType>::UniBESFind(values);
			//Filter<imDataType>::QsortOpt(values2);
			
//#define WHAT se->capacity/2
			if(se->capacity%2){							//odd
				dst[pos] = (unsigned)((unsigned)values[q1]
					+ 2*values[se->capacity/2] + values[q3])/4;

				/*if(values[WHAT] != values2[WHAT]){
					for(unsigned m=0;m < se->capacity; m++){
						cout << (int)values[m] << " ";
					}
					cout << "\n";
					for(unsigned m=0;m < se->capacity; m++){
						cout << (int)values2[m] << " ";
					}
					cout << "-----------\n";
				}*/
			}else{
				dst[pos] = (unsigned)((unsigned)values[q1] + values[se->capacity/2-1]
					+ values[se->capacity/2] + values[q3])/4;

				/*if(values[WHAT] != values2[WHAT]){
					for(unsigned m=0;m < se->capacity; m++){
						cout << (int)values[m] << " ";
					}
					cout << "\n";
					for(unsigned m=0;m < se->capacity; m++){
						cout << (int)values2[m] << " ";
					}
					cout << "------------\n";
				}*/
			}
		END_FOR3D;

			QueryPerformanceCounter(&stop);

		return (double)((stop.QuadPart-start.QuadPart)*1000)/frq.QuadPart;
	}
}

#endif