#include "SEManager.h"


template<typename imDataType>
bool SEManager<imDataType>::singletonFlag = 0;


template<typename imDataType>
SEManager<imDataType>* SEManager<imDataType>::Create()
{
	if(singletonFlag) return NULL;
 
	return new SEManager<imDataType>();
}

template<typename imDataType>
SEManager<imDataType>::SEManager(){

	int frameSize = GetFrameSize();
	int msize = 2*frameSize + 1;
	dictSize = msize*msize*msize;

	dictionary = new int[dictSize][3];

	int l=0;
	for(int i=0;i<frameSize; i++)
		for(int j=0;j<frameSize; j++)
			for(int k=0;k<frameSize; k++){
				dictionary[l][0] = j-2;
				dictionary[l][1] = k-2;
				dictionary[l][2] = i-2;
				l++;
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

3D image is organised in memory like in 3D image file    <<<<<---------!!!--------------!!!---!!!-------------

now - (1 + 2*dim[1] + 3*dim[1]*dim[2]) ... 2D image is classicaly 
in 1x2, therefore it's mask corresponds to middle square in SE 
definition
 
............+--- viewing direction for SE definition
............|......
..........+-|---+..
.........2..V../|..
......../...../.|..
.......+--1--+..|..
.......|.....|..+..
.......3.....|./...
.......|.....|/....
.......+-----+.....
...................
*/
template<typename imDataType>
int SEManager<imDataType>::Parse2SE(float *mask){
	
	structEl dummy;
	int last;
	int nonZero = 0;

	se.push_back(dummy);
	last = se.size - 1;
	
	//check number of non-zero elements
	for(int i=0; i<= last;i++){
		if(mask[i] != 0) nonZero++;
	}

	se[last].nbsize = nonZero;
	se[last].mask = new float[nonZero];
	se[last].nb = new int[nonZero];

	for(int i=0; i<= last;i++){
		if(mask[i] != 0) nonZero++;
	}




	

}