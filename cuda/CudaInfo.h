#pragma once
#include <cuda.h>
#include <Bacalar/structures.h>

/*

	Contains basic cuda information in inherited class

*/


class CudaInfo{

	static bool useCuda;

public:
	static bool EnableCuda(bool flag){return (useCuda = flag);} 
	static bool UseCuda(){return useCuda;}

};

bool CudaInfo::useCuda = false;