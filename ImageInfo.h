#pragma once
#include "Bacalar/ImageManager.h"

/*
	ImageInfo declared in ImageManager.h

	As nontemplate class, function definitions and static variable definitions goes here

	Setter functions include error check against loading multiple images with different dimmensions
*/

unsigned ImageInfo::imageDim = 0;
unsigned ImageInfo::imageDimensions[3] = {0,0,0};
unsigned ImageInfo::frameSize = 0;
unsigned ImageInfo::imageLineSize = 0; 
unsigned ImageInfo::imageSliceSize = 0;
unsigned ImageInfo::imageTotalPixelSize = 0;

bool ImageInfo::SetDim(int d){
	if(((imageDim != 0)&&(imageDim != d))||(d < 0)) return false;
	imageDim = d;
	return true;
}
bool ImageInfo::SetDimensions(int idx ,int dim){
	if(((imageDimensions[idx] != 0)&&(imageDimensions[idx] != dim))||(dim < 0)) return false;
	imageDimensions[idx] = dim;
	return true;
}
bool ImageInfo::SetFrameSize(int s){
	if(((frameSize != 0)&&(frameSize != s))||(s < 0)) return false;
	frameSize = s;
	return true;
}
bool ImageInfo::ComputeMetrics(){
	imageLineSize = imageDimensions[0]+2*frameSize;
	imageSliceSize = imageLineSize*(imageDimensions[1]+2*frameSize);

	imageTotalPixelSize = 1;
	for(unsigned i=0;i<imageDim;i++){ 	
		imageTotalPixelSize *= imageDimensions[i] + 2*frameSize;			//total image size (inc frame) 2D/3D
	}
	return true;
}