#pragma once

#include <string>
#include <vector>
using namespace std;
#include "ImageManager.h"

/*
	StructEl holds necessary information about used structure element
	- structEl.mask contains weights of wList pixels and is ignored in morphological 
		filters, so they stay fast despite StructEl.mask beeing float
	- wList[] contains only pre-computed pointer differences from cental pixel. Therefore 
		use it as follows:
			a = image[x + wihth*(y + z*height) + wList[i]]
		where [x,y,z] is processed (central) pixel, 'i' is in range [0-capacity] and 'image'
		is pointer to the image
	- mask orientation towards image (2D uses only one layer)

	...................................................................
	.0,0,0.+----+......................................................
	.....\/|.../|.<--- 3D image, mask has the same orientation.........
	...../\|../.|......................................................
	..../..+-/--+...+----+.<-- Top of the top (first in cfg file) layer
	.../.././../.../..../|.............................................
	..+----+../.../..../.|.............................................
	..|./..|./...+----+..+.<-- Bottom layer (rightmost in config file).
	..|/...|/....|....|./..............................................
	..+----+.....|....|/...............................................
	.............+----+................................................
	...................................................................

	- dictionary contains pointer differences for whole mask in the order
	  the mask is parsed by Parse2SE

*/


typedef struct _structEl{
	string name;
	int *wList;
	unsigned capacity;
	unsigned *mask;						//original input for later use (not used now)
} structEl;



template <typename imDataType>
class SEManager : private ImageInfo{

	static bool singletonFlag;
	int *dictionary;					//contains all pointer differences (entire neighbourhood)
	unsigned dictSize;
	vector<structEl*> se;
	SEManager();

public:

	static SEManager<imDataType>* Create();
	int GetSEIndex(string *name);
	structEl *GetSE(int index);
	bool DeleteAll();							//SE refresh - SEs still have to be parsed, so merging is worthless 
	int Parse2SE(string *name, unsigned *mask);	//float mask[dictsize]
	//int ReParse2PitchedSE(unsigned idx, unsigned pitch);
	bool SendToGpu();



};

#include "Bacalar/SEManagerCode.h"