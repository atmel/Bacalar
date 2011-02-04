#include "SEManager.h"


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

	for(int i=0; i< mSize; i++){
		for(int j=0; j< mSize*mSize; j++){
			cout << dictionary[j + i*mSize*mSize] << " ";
		}
		cout << "\n\n";
	}
	singletonFlag = 1;
}

/* 
	example 3x3x3 SE

	a b c | j k l | s t u
	d e f | m n o | v w x
	g h i | p q r | y z 1

	are pre-parsed line-by-line to an array of floats:

	a b c j k l s t u d e f ... 

	Pixels with mask value 0 will not be contained in final nb

	Mask topology is explained in SEManager.h. For clarification,
	'a' is in the top left corner of top layer, 's' 'corresponds'
	to 0,0,0 coordinates.


*/
template<typename imDataType>
int SEManager<imDataType>::Parse2SE(string *name, float *mask){
	
	structEl dummy;
	int justAdded;
	int nonZero = 0;

	se.push_back(dummy);
	justAdded = se.size() - 1;
	
	//check number of non-zero elements in mask (parsed to line of floats)
	for(int i=0; i < dictSize;i++){
		if(mask[i] != 0.0) nonZero++;
	}

	se[justAdded].nbSize = nonZero;
	se[justAdded].mask = new float[nonZero];
	se[justAdded].nb = new int[nonZero];
	se[justAdded].name = *name;

	//fill SE mask - only used pixels
	nonZero = 0;		//as count variable
	for(int i=0; i < dictSize;i++){		
		if(mask[i] != 0.0){
			se[justAdded].mask[nonZero] = mask[i];
			se[justAdded].nb[nonZero] = dictionary[i];
			nonZero++;
		}
	}
}