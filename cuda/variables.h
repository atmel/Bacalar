#ifndef CUDA_VARS
#define CUDA_VARS

#include <cuda.h>

#define MAX_SE 100

//SEs
__constant__ unsigned *gpuWeightedList[MAX_SE];
__constant__ unsigned gpuCap[MAX_SE];
//__constant__ float *gpuMask[MAX_SE];

//filter additionals computed from SE data
//__constant__ unsigned gpuMedianSortArrSize[MAX_SE];
//__constant__ unsigned gpuBESSortArrSize[MAX_SE];


//metrics
__constant__ unsigned gpuImageDim, gpuImageDimensions[3], gpuFrameSize;
__constant__ unsigned gpuImageLineSize, gpuImageSliceSize, gpuImageTotalPixelSize;	//inluding frame
__constant__ unsigned gpuImageWidth, gpuImageSliceArea, gpuImageSize;				//without frame

//textures

texture<unsigned char> uchar1DTextRef;

#endif