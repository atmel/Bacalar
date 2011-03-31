#pragma once

/*

	Contains morphology filters (not using mask values - treating them as 1/0)

*/

#include "Bacalar/Filter.h"




template <typename imDataType>
float Filter<imDataType>::Erode(imDataType* dst, int seIndex, imDataType* srcA, fourthParam<imDataType> p4){
	imDataType min;
	structEl *se = sem->GetSE(seIndex);
	memset(dst,0,GetTotalPixelSize()*sizeof(imDataType));
	unsigned long pos;

	//3D
	BEGIN_FOR3D(pos)			
		min = srcA[pos + se->nb[0]];		//find minimum
		for(unsigned m=1;m < se->nbSize; m++){
			if(min > srcA[pos + se->nb[m]])
				min = srcA[pos + se->nb[m]];
		}
		dst[pos] = min;
	END_FOR3D;
	return 1;
}


template <typename imDataType>
float Filter<imDataType>::Dilatate(imDataType* dst, int seIndex, imDataType* srcA, fourthParam<imDataType> p4){
	imDataType max;
	structEl *se = sem->GetSE(seIndex);
	memset(dst,0,size*sizeof(imDataType));
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

}
template <typename imDataType>
float Filter<imDataType>::Close (imDataType* dst, int seIndex, imDataType* srcA, fourthParam<imDataType> p4){

}
