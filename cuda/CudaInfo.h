#pragma once
#include <cuda.h>
#include <Bacalar/structures.h>

/*

	Contains basic cuda information in inherited class

*/


class CudaInfo{

	static bool useCuda;

public:

	bool UseCuda(){return useCuda;}

};