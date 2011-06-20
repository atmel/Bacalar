#include "SEManager.h"
#include "Bacalar/cuda/variables.h"


#include <iostream>
using namespace std;

bool SEManager::singletonFlag = 0;

/*
	Create function handles singleton policy
*/


SEManager* SEManager::Create()
{
	if(singletonFlag) return NULL;
 
	return new SEManager();
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

SEManager::SEManager(){

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

	singletonFlag = 1;
}

/* 
	example 3x3x3 SE

	a, b, c,  j, k, l,  s, t, u,
	d, e, f,  m, n, o,  v, w, x,
	g, h, i,  p, q, r,  y, z, 1,

	are pre-parsed line-by-line to an array of floats:

	a b c j k l s t u d e f ... 

	Pixels with mask value 0 will not be contained in final wList

	Mask topology is explained in SEManager.h. For clarification,
	'a' is in the top left corner of top layer, 's' 'corresponds'
	to 0,0,0 coordinates -- see ImageManager.h.

*/

int SEManager::Parse2SE(string *name, unsigned *mask){
	
	int justAdded;

	se.push_back(new structEl);
	justAdded = se.size() - 1;
	
	//compute capacity of mask and copy the mask
	se[justAdded]->capacity = 0;
	se[justAdded]->mask = new unsigned[dictSize];
	for(unsigned i=0; i < dictSize;i++){
		se[justAdded]->capacity += mask[i];
		se[justAdded]->mask[i] = mask[i];
	}

	se[justAdded]->wList = new int[se[justAdded]->capacity];
	se[justAdded]->name = *name;

	//fill wList - voxels with weight 0 will not be included
	int idx=0;
	for(unsigned i=0; i < dictSize;i++){		
		for(unsigned j=0;j < mask[i];j++){
			se[justAdded]->wList[idx] = dictionary[i];
			idx++;
		}
	}

	for(unsigned i =0;i<se[justAdded]->capacity;i++){
		cout << (signed int)se[justAdded]->wList[i] << " ";
	}
	cout << "\nSE size: " << se[justAdded]->capacity << '\n';
	return se[justAdded]->capacity;
}




structEl* SEManager::GetSE(int index){
	return se[index];
}


bool SEManager::SendToGpu(){
	
	unsigned capacities[MAX_SE];//, medSizes[MAX_SE], BESSizes[MAX_SE];
	for(unsigned i=0;i<se.size();i++){
		capacities[i] = se[i]->capacity;
			//compute filter additionals
		//medSizes[i] = capacities[i]/2+2;			//it's faster to do that on GPU
		//BESSizes[i] = (3*capacities[i])/4+2;
	}
	cudaMemcpyToSymbol(gpuCap, capacities, sizeof(unsigned)*se.size());
	//cudaMemcpyToSymbol(gpuMedianSortArrSize, medSizes, sizeof(unsigned)*se.size());
	//cudaMemcpyToSymbol(gpuBESSortArrSize, BESSizes, sizeof(unsigned)*se.size());
	cout << "SEManager, send cuda error:" << cudaGetErrorString(cudaGetLastError()) << '\n';
	unsigned *wLists[MAX_SE];
	//float *masks[MAX_SE];
	for(unsigned i=0;i<se.size();i++){
		cudaMalloc(&(wLists[i]),sizeof(unsigned)*se[i]->capacity);
		//cout << "SEManager, send cuda error:" << cudaGetErrorString(cudaGetLastError()) << '\n';
		cudaMemcpy((void*)(wLists[i]),se[i]->wList,sizeof(unsigned)*se[i]->capacity,cudaMemcpyHostToDevice);
		//cout << "SEManager, send cuda error:" << cudaGetErrorString(cudaGetLastError()) << '\n';
//		cudaMalloc(&(masks[i]),sizeof(float)*se[i]->capacity);
		//cout << "SEManager, send cuda error:" << cudaGetErrorString(cudaGetLastError()) << '\n';
//		cudaMemcpy((void*)(masks[i]),se[i]->weight,sizeof(float)*se[i]->capacity,cudaMemcpyHostToDevice);
		//cout << "SEManager, send cuda error:" << cudaGetErrorString(cudaGetLastError()) << '\n';
	}
	cudaMemcpyToSymbol(gpuWeightedList, wLists, sizeof(unsigned*)*se.size());
	//cout << "SEManager, send cuda error:" << cudaGetErrorString(cudaGetLastError()) << '\n';
//	cudaMemcpyToSymbol(gpuMask, masks, sizeof(float*)*se.size());
	cout << "SEManager, send cuda error:" << cudaGetErrorString(cudaGetLastError()) << '\n';
	return true;
}