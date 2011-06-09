#pragma once

#ifndef FILTER_CLASS
#define FILTER_CLASS

//#include <CUDAEnviroment header>
#include <Bacalar/structures.h>

/*

	Abstract Filter Class

	contains all fiter functions, supporting functions and
	necessary static variables.
	It is used as abstract function container - functions are called
	from lists via pointers.

*/ 

#include "Bacalar/SEManager.h"
#include "Bacalar/ImageManager.h"
#include "Bacalar/structures.h"		//for basic definitions
#include "Bacalar/cuda/CudaInfo.h"

template <typename imDataType>
union fourthParam{
	imDataType *srcB;
	int k;
	struct _int2{
		int k,j;
	} int2;
	float f;
};

template <typename imDataType>
class Filter : private ImageInfo, private CudaInfo{

	static unsigned long sliceUpperBound, lineUpperBound;	//for fast FORxD macro
	static SEManager<imDataType> *sem;

	//supporting functions (CPU):
	inline imDataType Min(imDataType x, imDataType y)
		{return x>y?y:x;}
	inline imDataType Max(imDataType x, imDataType y)
		{return x<y?y:x;}

public:static bool QsortOpt(imDataType *base, unsigned initBaseLength = 0); 
	   static imDataType MedianFindOpt(imDataType *base, unsigned initBaseLength = 0); 
	   static imDataType Forgetful(imDataType *sortArr, unsigned initBaseLength);

public:

	static bool Init(SEManager<imDataType> *_sem);							//initialize static variables

	//CPU
	//basic operations
	static float Add (imDataType* dst, int seIndex, imDataType* srcA, fourthParam<imDataType> p4);		//A+B
	static float ASubB (imDataType* dst, int seIndex, imDataType* srcA, fourthParam<imDataType> p4);	//A-B
	static float AddNoise (imDataType* dst, int seIndex, imDataType* srcA, fourthParam<imDataType> p4);	//A-B
	//morphology
	static float Erode (imDataType* dst, int seIndex, imDataType* srcA, fourthParam<imDataType> p4);	//er(A)
	static float Dilatate (imDataType* dst, int seIndex, imDataType* srcA, fourthParam<imDataType> p4);	//dil(A)
	static float Open (imDataType* dst, int seIndex, imDataType* srcA, fourthParam<imDataType> p4);		//op(A)
	static float Close (imDataType* dst, int seIndex, imDataType* srcA, fourthParam<imDataType> p4);	//
	static float WTH (imDataType* dst, int seIndex, imDataType* srcA, fourthParam<imDataType> p4);		//A-
	static float Edge (imDataType* dst, int seIndex, imDataType* srcA, fourthParam<imDataType> p4);	//er(A)
	//sorted list filters - use mask
	static float Median (imDataType* dst, int seIndex, imDataType* srcA, fourthParam<imDataType> p4);
	static float WMedian (imDataType* dst, int seIndex, imDataType* srcA, fourthParam<imDataType> p4);
	static float KthQuantil (imDataType* dst, int seIndex, imDataType* srcA, fourthParam<imDataType> p4);
	static float KSubJQuantil (imDataType* dst, int seIndex, imDataType* srcA, fourthParam<imDataType> p4);
	static float BES (imDataType* dst, int seIndex, imDataType* srcA, fourthParam<imDataType> p4);
	static float WBES (imDataType* dst, int seIndex, imDataType* srcA, fourthParam<imDataType> p4);

};

/*

	Useful macros ---------------------------------

*/

#define BEGIN_FOR3D(pos)\
for(unsigned long col,ln,lnAndSl,sl=GetFrameSize()*GetSliceSize(); sl<sliceUpperBound; sl+=GetSliceSize())\
for(ln=GetFrameSize()*GetLineSize(); ln<lineUpperBound; ln+=GetLineSize()){\
lnAndSl=ln+sl;\
for(col=GetFrameSize();col<GetDimensions(0)+GetFrameSize();col++){\
pos=lnAndSl+col;

#define END_FOR3D }}




#include "Bacalar/filter/kernels.h"
#include "Bacalar/filter/kernels/kerMorph.h"
#include "Bacalar/filter/kernels/kerSorted.h"
#include "Bacalar/filter/kernels/kerWalsh.h"
#include "Bacalar/filter/InitandAlgebra.h"
#include "Bacalar/filter/Morphology.h"
#include "Bacalar/filter/SortedList.h"
#include "Bacalar/filter/Walschlist.h"
#include "Bacalar/filter/fastsort.h"

#endif

