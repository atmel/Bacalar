#pragma once

#include "Bacalar/Filter.h"



template <typename imDataType>
float Filter<imDataType>::Median(imDataType* dst, int seIndex, imDataType* srcA, fourthParam<imDataType> p4){
	imDataType *values;
	structEl *se = sem->GetSE(seIndex);
	memset(dst,0,size*sizeof(imDataType));
	values = new imDataType[se->nbSize];
	Filter<imDataType>::QsortOpt(NULL,se->nbSize);
	unsigned i,j,k;

	//3D
	FOR3D(i,j,k){			
		for(unsigned m=1;m < se->nbSize; m++){
			values[m] = srcA[k + j*lineSize + i*sliceSize + se->nb[m]];
		}
		Filter<imDataType>::QsortOpt(values);
		if(se->nbSize%2){							//odd
			dst[k + j*lineSize + i*sliceSize] = values[se->nbSize/2];
		}else{
			dst[k + j*lineSize + i*sliceSize] = (values[se->nbSize/2]+values[se->nbSize/2+1])/2;
		}
	}
	return 1;
}