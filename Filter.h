#pragma once

//#include <CUDAEnviroment header>
#include <Bacalar/structures.h>

/*

Abstract Filter Class

contains all fiter functions, supporting functions and
necessary static variables.
It is used as function container - functions are called
from lists via pointers.

*/ 

template <typename imDataType>
union fourthParam{
	imDataType srcB;
	int k;
	struct _kj{
		int k,j;
	} kj;
};


template <typename imDataType>
class Filter{

	static int imageDim;						//2D, 3D
	static int imageDimensions[3];				//in pixels
	static int frameSize;						//width in pixels
	//static float CPU[XY];						//array with function pointers for easy access
	//static float GPU[XY];

//supporting functions (CPU):
	inline imDataType Min(imDataType x, imDataType y)
		{return x>y?y:x}
	inline imDataType Max(imDataType x, imDataType y)
		{return x<y?y:x}

public:

//CPU
	//basic operations
	static float Add (imDataType* dst, int seIndex, imDataType* srcA, fourthParam p4);		//A+B
	static float Sub (imDataType* dst, int seIndex, imDataType* srcA, fourthParam p4);		//A-B
	//morphology
	static float Erode (imDataType* dst, int seIndex, imDataType* srcA, fourthParam p4);	//er(A)
	static float Dilatate (imDataType* dst, int seIndex, imDataType* srcA, fourthParam p4);	//dil(A)
	static float Open (imDataType* dst, int seIndex, imDataType* srcA, fourthParam p4);		//op(A)
	static float Close (imDataType* dst, int seIndex, imDataType* srcA, fourthParam p4);	//A-B
	static float WTH (imDataType* dst, int seIndex, imDataType* srcA, fourthParam p4);		//A-
	//sorted list filters - use mask
	static float Median (imDataType* dst, int seIndex, imDataType* srcA, fourthParam p4);
	static float Wmedian (imDataType* dst, int seIndex, imDataType* srcA, fourthParam p4);
	static float KthValue (imDataType* dst, int seIndex, imDataType* srcA, fourthParam p4);
	static float KthSubJth (imDataType* dst, int seIndex, imDataType* srcA, fourthParam p4);
	static float BES (imDataType* dst, int seIndex, imDataType* srcA, fourthParam p4);
	static float WBES (imDataType* dst, int seIndex, imDataType* srcA, fourthParam p4);

};