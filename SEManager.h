#pragma once

#include <string>
#include <vector>
using namespace std;
#include "ImageManager.h"

/*
mask is ignored in morphological filters, so they stay fast despite mask uses floats
nb[] contains only pre-computed pointer differences from cental pixel. Therefore use 
it as follows:
	*image;
	*(image + x + wihth*(y + z*height) + nb[xy]) = ...

*/


typedef struct _structEl{
	string name;
	int *nb;
	int nbsize;
	float *mask;
} structEl;

template <typename imDataType>
class SEManager : private ImageInfo{

	static bool singletonFlag;
	int **dictionary;					//3 * SE size
	int dictSize;
	vector<structEl> se;
	SEManager();

public:

	static SEManager<imDataType>* Create();
	structEl *GetSE(int index);
	int Parse2SE(float *mask);			//float [dictsize]



};