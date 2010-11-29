#pragma once

#include <string>
#include <vector>
using namespace std;

/*
mask is ignored in morphological filters, so they stay fast despite mask uses floats
nb[] contains only pre-computed pointer differences from cental pixel. Therefore use 
it as follows:
	*image;
	*(image + x + wihth*(y + z*height) + nb[xy]) = ...

*/


typedef struct{
	string name;
	int *nb;
	int nbsize;
	float *mask;
} structEl;

template <typename imDataType>
class SEManager{

	int **dictionary;					//3 * SE size

	vector<structEl> se;

public:

	static bool Create(int framesize);
	structEl *GetSE(int index);
	int Parse2SE(string istr);			//3 lines are pre-parsed to single line



};