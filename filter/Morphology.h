#pragma once

/*

	Contains morphology filters (not using mask values - treating them as 1/0)

*/

#include "Bacalar/Filter.h"




template <typename imDataType>
float Filter<imDataType>::Erode(imDataType* dst, int seIndex, imDataType* srcA, fourthParam<imDataType> p4){
	imDataType min;
	structEl *se = sem->GetSE(seIndex);
	memset(dst,0,size*sizeof(imDataType));
	unsigned i,j,k;

	//3D
	FOR3D(i,j,k){			
		min = srcA[k + j*lineSize + i*sliceSize + se->nb[0]];		//find maximum
		for(unsigned m=1;m < se->nbSize; m++){
			if(min > srcA[k + j*lineSize + i*sliceSize + se->nb[m]])
				min = srcA[k + j*lineSize + i*sliceSize + se->nb[m]];
		}
		dst[k + j*lineSize + i*sliceSize] = min;
	}
	return 1;
}


template <typename imDataType>
float Filter<imDataType>::Dilatate(imDataType* dst, int seIndex, imDataType* srcA, fourthParam<imDataType> p4){
	imDataType max;
	structEl *se = sem->GetSE(seIndex);
	memset(dst,0,size*sizeof(imDataType));
	unsigned i,j,k;

	//3D
	FOR3D(i,j,k){			
		min = srcA[k + j*lineSize + i*sliceSize + se->nb[0]];		//find maximum
		for(unsigned m=1;m < se->nbSize; m++){
			if(max < srcA[k + j*lineSize + i*sliceSize + se->nb[m]])
				max = srcA[k + j*lineSize + i*sliceSize + se->nb[m]];
		}
		dst[k + j*lineSize + i*sliceSize] = max;
	}
	return 1;
}


template <typename imDataType>
float Filter<imDataType>::Open (imDataType* dst, int seIndex, imDataType* srcA, fourthParam<imDataType> p4){

}
template <typename imDataType>
float Filter<imDataType>::Close (imDataType* dst, int seIndex, imDataType* srcA, fourthParam<imDataType> p4){

}
