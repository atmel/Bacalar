#pragma once

#include "Bacalar/Filter.h"

template <typename imDataType>
unsigned Filter<imDataType>::imageDim = 0;

template <typename imDataType>
unsigned Filter<imDataType>::imageDimensions[3] = {0,0,0};	

template <typename imDataType>
unsigned Filter<imDataType>::frameSize = 0;

template <typename imDataType>
unsigned Filter<imDataType>::lineSize = 0; 

template <typename imDataType>
unsigned Filter<imDataType>::sliceSize = 0;

template <typename imDataType>
unsigned long Filter<imDataType>::size = 0;

template <typename imDataType>
SEManager<imDataType>* Filter<imDataType>::sem = NULL;

template <typename imDataType>
bool Filter<imDataType>::Init(SEManager<imDataType> *_sem){
	imageDim = ImageInfo::GetDim();							
	frameSize = ImageInfo::GetFrameSize();

	size = 1;
	for(unsigned i=0;i<imageDim;i++){ 
		imageDimensions[i] = ImageInfo::GetDimensions(i);	
		size *= imageDimensions[i] + 2*frameSize;			//total image size (inc frame) 2D/3D
	}
	sem = _sem;

	lineSize = imageDimensions[0]+2*frameSize;
	if(imageDim == 3) sliceSize = lineSize*(imageDimensions[1]+2*frameSize);

	return true;
}		





template <typename imDataType>
float Filter<imDataType>::Erode (imDataType* dst, int seIndex, imDataType* srcA, fourthParam<imDataType> p4){
	imDataType min;
	structEl *se = sem->GetSE(seIndex);
	memset(dst,0,size*sizeof(imDataType));

	//3D
	for(unsigned i=frameSize;i<imageDimensions[2]+frameSize;i++)			//slowest (up-down)
		for(unsigned j=frameSize;j<imageDimensions[1]+frameSize;j++)
			for(unsigned k=frameSize;k<imageDimensions[0]+frameSize;k++){	//fastest (left-right)
				min = srcA[k + j*lineSize + i*sliceSize + se->nb[0]];		//find minimum
				for(unsigned m=1;m < se->nbSize; m++){
					if(min > srcA[k + j*lineSize + i*sliceSize + se->nb[m]])
						min = srcA[k + j*lineSize + i*sliceSize + se->nb[m]];
				}
				dst[k + j*lineSize + i*sliceSize] = min;
			}
	return 1;
}
template <typename imDataType>
float Filter<imDataType>::Dilatate (imDataType* dst, int seIndex, imDataType* srcA, fourthParam<imDataType> p4){

}
template <typename imDataType>
float Filter<imDataType>::Open (imDataType* dst, int seIndex, imDataType* srcA, fourthParam<imDataType> p4){

}
template <typename imDataType>
float Filter<imDataType>::Close (imDataType* dst, int seIndex, imDataType* srcA, fourthParam<imDataType> p4){

}
