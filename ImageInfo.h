#pragma once
#include "Bacalar/ImageManager.h"

/*
	ImageInfo declared in ImageManager.h

	As nontemplate class, function definitions and static variable definitions goes here

	Setter functions include error check against loading multiple images with different dimmensions
*/

int ImageInfo::imageDim = 0;
int ImageInfo::imageDimensions[3] = {0,0,0};
int ImageInfo::frameSize = 0;

bool ImageInfo::SetDim(int d){
	if((imageDim != 0)&&(imageDim != d)) return false;
	imageDim = d;
	return true;
}
bool ImageInfo::SetDimensions(int idx ,int dim){
	if((imageDimensions[idx] != 0)&&(imageDimensions[idx] != dim)) return false;
	imageDimensions[idx] = dim;
	return true;
}
bool ImageInfo::SetFrameSize(int s){
	if((frameSize != 0)&&(frameSize != s)) return false;
	frameSize = s;
	return true;
}

int ImageInfo::GetDim(){return imageDim;}
int ImageInfo::GetDimensions(int index){return imageDimensions[index];}
int ImageInfo::GetFrameSize(){return frameSize;}