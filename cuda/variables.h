#ifndef CUDA_VARS
#define CUDA_VARS

#include <cuda.h>

#define MAX_SE 100

//SEs
__constant__ unsigned *gpuNb[MAX_SE];
__constant__ unsigned gpuNbSize[MAX_SE];
__constant__ float *gpuMask[MAX_SE];

//metrics
__constant__ unsigned gpuImageDim, gpuImageDimensions[3], gpuFrameSize;
__constant__ unsigned gpuImageLineSize, gpuImageSliceSize, gpuImageTotalPixelSize;	//inluding frame
__constant__ unsigned gpuImageWidth, gpuImageSliceArea, gpuImageSize;				//without frame

//textures

texture<unsigned char> uchar1DTextRef;

#endif