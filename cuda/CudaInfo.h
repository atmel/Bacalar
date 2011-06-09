#pragma once
#include <cuda.h>
#include <Bacalar/structures.h>

/*

	Contains basic cuda information in inherited class

*/


class CudaInfo{

	static bool useCuda;
	//static bool pitchedMemory;						//if GPU image sould be allocated as pitched 3D

public:
	static bool EnableCuda(bool flag){return (useCuda = flag);} 
	//static bool EnableCudaPitchedMem(bool flag){
	//	if(pitchedMemory == true) return false;		//can be only set prior CUDA acces to GPU memory
	//	pitchedMemory = flag;
	//	return true;
	//} 
	static bool UseCuda(){return useCuda;}

};

bool CudaInfo::useCuda = false;
//bool CudaInfo::pitchedMemory = false;