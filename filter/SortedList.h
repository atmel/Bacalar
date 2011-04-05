#pragma once

#ifndef SORTED_LIST
#define SORTED_LIST

#include "Bacalar/Filter.h"



template <typename imDataType>
float Filter<imDataType>::Median(imDataType* dst, int seIndex, imDataType* srcA, fourthParam<imDataType> p4){
	imDataType *values;
	structEl *se = sem->GetSE(seIndex);
	memset(dst,0,GetTotalPixelSize()*sizeof(imDataType));
	values = new imDataType[se->nbSize];
	Filter<imDataType>::QsortOpt(NULL,se->nbSize);		//initialize qsort
	unsigned long pos;

	//3D
	BEGIN_FOR3D(pos)		
		for(unsigned m=1;m < se->nbSize; m++){
			values[m] = srcA[pos + se->nb[m]];
		}
		Filter<imDataType>::QsortOpt(values);
		if(se->nbSize%2){							//odd
			dst[pos] = values[se->nbSize/2];
		}else{
			dst[pos] = (values[se->nbSize/2]+values[se->nbSize/2+1])/2;
		}
	END_FOR3D;

	return 1;
}

#endif