#include <cuda.h>
#include <iostream>
using namespace std;



__global__ void VAdd(int* A, int* B, int* C, int size)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx < size) C[idx] = A[idx] + B[idx];
}

__global__ void ArrInit(int *arr, int size)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx < size) arr[idx] = idx;
}



#define SIZE 10000

int main(int argc, char* argv[]){

	int *a, *b, *c;

	a = new int[SIZE];
	c = new int[SIZE];

	CUdeviceptr *gpuA, *gpuB, *gpuC;

	CUdevprop props;
	memset(&props,0,sizeof(props));

	cuInit(0);
	cuDeviceGetProperties(&props,0);

	cout << "clockRate: " << (int)props.clockRate << '\n';
	cout << "maxGridSize: "<< (int)props.maxGridSize[0]<< '\n'<< (int)props.maxGridSize[1]<< '\n'<< (int)props.maxGridSize[2]<< '\n';
	cout << "maxThreadsDim: "<< (int)props.maxThreadsDim[0] << '\n'<< (int)props.maxThreadsDim[1] << '\n'<< (int)props.maxThreadsDim[2] << '\n';
	cout << "maxThreadsPerBlock: "<< (int)props.maxThreadsPerBlock << '\n';
	cout << "memPitch: "<< (int)props.memPitch << '\n';
	cout << "regsPerBlock: "<< (int)props.regsPerBlock << '\n';
	cout << "sharedMemPerBlock: "<< (int)props.sharedMemPerBlock << '\n';
	cout << "SIMDWidth: "<< (int)props.SIMDWidth << '\n';
	cout << "textureAlign: "<< (int)props.textureAlign << '\n';
	cout << "totalConstantMemory: "<< (int)props.totalConstantMemory << '\n';

	size_t size = sizeof(int)*SIZE;
	cuMemAlloc(gpuA,size);
	cuMemAlloc(gpuB,size);
	cuMemAlloc(gpuC,size);
	

	ArrInit<<<20,512>>>((int*)gpuA,SIZE);
	ArrInit<<<20,512>>>((int*)gpuB,SIZE);
	VAdd<<<20,512>>>((int*)gpuA,(int*)gpuB,(int*)gpuC,SIZE);

	cuMemcpyDtoH((void*)c,*gpuC,size);

	for(int i=0;i<100;i++) cout << c[i] << ", ";


return 0;
}