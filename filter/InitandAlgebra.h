#pragma once

/*

	Must be included first in Filter.h, contains declarations of static variables

*/

#include "Bacalar/Filter.h"	
#include <math.h>

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
	

//float specialization
template <>
float Filter<float>::Add (float* dst, int seIndex, float* srcA, fourthParam<float> p4){

	float val;
	unsigned i,j,k;
	//3D

	FOR3D(i,j,k){
		if(srcA[k + j*lineSize + i*sliceSize]+p4.srcB[k + j*lineSize + i*sliceSize] > FLOAT_TRUE){
			val = FLOAT_TRUE;
		}else{
			val = srcA[k + j*lineSize + i*sliceSize]+p4.srcB[k + j*lineSize + i*sliceSize];
		}
		dst[k + j*lineSize + i*sliceSize] = val;
	}
	return 1;
}	

template <>
float Filter<unsigned int>::Add (unsigned int* dst, int seIndex, unsigned int* srcA, fourthParam<unsigned int> p4){

	unsigned int val;
	unsigned i,j,k;
	//3D

	FOR3D(i,j,k){
		if(srcA[k + j*lineSize + i*sliceSize]/2+p4.srcB[k + j*lineSize + i*sliceSize]/2 >= INT_TRUE/2){
			val = INT_TRUE;
		}else{
			val = srcA[k + j*lineSize + i*sliceSize]+p4.srcB[k + j*lineSize + i*sliceSize];
		}
		dst[k + j*lineSize + i*sliceSize] = val;
	}
	return 1;
}	

template <>
float Filter<unsigned char>::Add (unsigned char* dst, int seIndex, unsigned char* srcA, fourthParam<unsigned char> p4){

	unsigned char val;
	unsigned i,j,k;
	//3D

	FOR3D(i,j,k){
		if(srcA[k + j*lineSize + i*sliceSize]/2+p4.srcB[k + j*lineSize + i*sliceSize]/2 >= CHAR_TRUE/2){
			val = CHAR_TRUE;
		}else{
			val = srcA[k + j*lineSize + i*sliceSize]+p4.srcB[k + j*lineSize + i*sliceSize];
		}
		dst[k + j*lineSize + i*sliceSize] = val;
	}
	return 1;
}	


template <typename imDataType>		//A-B (lukasiewicz A and B)
float Filter<imDataType>::ASubB (imDataType* dst, int seIndex, imDataType* srcA, fourthParam<imDataType> p4){

	unsigned i,j,k;
	//3D

	FOR3D(i,j,k){		
		if(srcA[k + j*lineSize + i*sliceSize] < p4.srcB[k + j*lineSize + i*sliceSize]){
			dst[k + j*lineSize + i*sliceSize] = 0;
		}else{
			dst[k + j*lineSize + i*sliceSize] = srcA[k + j*lineSize + i*sliceSize] -
				p4.srcB[k + j*lineSize + i*sliceSize];
		}
	}
	return 1;
}	

/*

	Adds random noise to image

	p4.k is in range 0-1000, determining noise level 0-<TYPE>_TRUE

*/

template <typename imDataType>
float Filter<imDataType>::AddNoise (imDataType* dst, int seIndex, imDataType* srcA, fourthParam<imDataType> p4){

	rand
}
