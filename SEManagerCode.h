#include "SEManager.h"
#include "Bacalar/cuda/variables.h"


#include <iostream>
using namespace std;
template<typename imDataType>
bool SEManager<imDataType>::singletonFlag = 0;

/*
	Create function handles singleton policy
*/
template<typename imDataType>
SEManager<imDataType>* SEManager<imDataType>::Create()
{
	if(singletonFlag) return NULL;
 
	return new SEManager<imDataType>();
}
/*
	Private contructor of SEManager
	- frame size needs to be known before class instantiation
	- image size must be also known before instantiation
	- dimension must be also known (2D mask has only one layer and
		therefore different parsing and dictionary)
	- dictionary translates position in string passed to the Parse2SE
		function into pointer differences within image (dimensions
		are known, so absolute differences can be calculated)
*/
template<typename imDataType>
SEManager<imDataType>::SEManager(){

	if(!GetDimensions(0)) return;			//no image loaded

	unsigned int frame = GetFrameSize();
	int mSize = 2*frame + 1;
	unsigned int lineSize = GetDimensions(0)+2*frame;
	unsigned int sliceSize = lineSize*(GetDimensions(1)+2*frame);
	dictSize = mSize*mSize*mSize;

	dictionary = new int[dictSize];

	for(int i=0;i<mSize; i++)
		for(int j=0;j<mSize; j++)
			for(int k=0;k<mSize; k++){
				dictionary[k+j*mSize+i*mSize*mSize] = 
					k-frame + (frame-j)*sliceSize + (i-frame)*lineSize;
			}
	
	se.clear();

	/*for(int i=0; i< mSize; i++){
		for(int j=0; j< mSize*mSize; j++){
			cout << dictionary[j + i*mSize*mSize] << " ";
		}
		cout << "\n\n";
	}*/
	singletonFlag = 1;
}

/* 
	example 3x3x3 SE

	a, b, c,  j, k, l,  s, t, u,
	d, e, f,  m, n, o,  v, w, x,
	g, h, i,  p, q, r,  y, z, 1,

	are pre-parsed line-by-line to an array of floats:

	a b c j k l s t u d e f ... 

	Pixels with mask value 0 will not be contained in final nb

	Mask topology is explained in SEManager.h. For clarification,
	'a' is in the top left corner of top layer, 's' 'corresponds'
	to 0,0,0 coordinates.


*/
template<typename imDataType>
int SEManager<imDataType>::Parse2SE(string *name, float *mask){
	
	//structEl dummy;
	int justAdded;
	int nonZero = 0;

	se.push_back(new structEl);
	justAdded = se.size() - 1;
	
	//check number of non-zero elements in mask (parsed to line of floats)
	for(int i=0; i < dictSize;i++){
		if(mask[i] != 0.0) nonZero++;
	}

	se[justAdded]->nbSize = nonZero;
	se[justAdded]->mask = new float[nonZero];
	se[justAdded]->origInput = new float[dictSize];
	se[justAdded]->nb = new unsigned[nonZero];
	se[justAdded]->name = *name;

	for(int i=0; i < dictSize;i++) se[justAdded]->origInput[i] = mask[i];

	//fill SE mask - only used pixels
	nonZero = 0;		//as count variable
	for(int i=0; i < dictSize;i++){		
		if(mask[i] != 0.0){
			se[justAdded]->mask[nonZero] = mask[i];
			se[justAdded]->nb[nonZero] = dictionary[i];
			nonZero++;
		}
	}
	cout << "SE size: " << se[justAdded]->nbSize << '\n';
	return se[justAdded]->nbSize;
}



template<typename imDataType>
structEl* SEManager<imDataType>::GetSE(int index){
	return se[index];
}

template<typename imDataType>
bool SEManager<imDataType>::SendToGpu(){
	
	unsigned sizes[MAX_SE], medSizes[MAX_SE], BESSizes[MAX_SE];
	for(unsigned i=0;i<se.size();i++){
		sizes[i] = se[i]->nbSize;
			//compute filter additionals
		medSizes[i] = sizes[i]/2+2;
		BESSizes[i] = (3*sizes[i])/4+2;
	}
	cudaMemcpyToSymbol(gpuNbSize, sizes, sizeof(unsigned)*se.size());
	cudaMemcpyToSymbol(gpuMedianSortArrSize, medSizes, sizeof(unsigned)*se.size());
	cudaMemcpyToSymbol(gpuBESSortArrSize, BESSizes, sizeof(unsigned)*se.size());
	cout << "SEManager, send cuda error:" << cudaGetErrorString(cudaGetLastError()) << '\n';
	unsigned *nbs[MAX_SE];
	float *masks[MAX_SE];
	for(unsigned i=0;i<se.size();i++){
		cudaMalloc(&(nbs[i]),sizeof(unsigned)*se[i]->nbSize);
		//cout << "SEManager, send cuda error:" << cudaGetErrorString(cudaGetLastError()) << '\n';
		cudaMemcpy((void*)(nbs[i]),se[i]->nb,sizeof(unsigned)*se[i]->nbSize,cudaMemcpyHostToDevice);
		//cout << "SEManager, send cuda error:" << cudaGetErrorString(cudaGetLastError()) << '\n';
		cudaMalloc(&(masks[i]),sizeof(float)*se[i]->nbSize);
		//cout << "SEManager, send cuda error:" << cudaGetErrorString(cudaGetLastError()) << '\n';
		cudaMemcpy((void*)(masks[i]),se[i]->mask,sizeof(float)*se[i]->nbSize,cudaMemcpyHostToDevice);
		//cout << "SEManager, send cuda error:" << cudaGetErrorString(cudaGetLastError()) << '\n';
	}
	cudaMemcpyToSymbol(gpuNb, nbs, sizeof(unsigned*)*se.size());
	//cout << "SEManager, send cuda error:" << cudaGetErrorString(cudaGetLastError()) << '\n';
	cudaMemcpyToSymbol(gpuMask, masks, sizeof(float*)*se.size());
	cout << "SEManager, send cuda error:" << cudaGetErrorString(cudaGetLastError()) << '\n';
	return true;
}