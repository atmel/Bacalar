#pragma once

/*

	File topology:
	
	.............Dim 0 (fastest).......
	.............--->..................
	............+--------+.............
	..Dim 1.././|......./|.|..Dim 2....
	.......././.|....../.|.|.(slowest).
	......|/./..|0,0,0/..|.V...........
	......../...+----/---+.............
	......./.../..../.../..............
	......+--------+.../...............
	......|../.....|../................
	......|./......|./.................
	......|/.......|/..................
	......+--------+...................
	........./.........................
	......../..........................
	......|/..Eyes face this way.......
	...................................

	0,0,0 pixel corresponds to bottom left pixel of transformed 2D image
	(it is actually printed upside down as data are read from 0,0,0 -> top left)


*/

#include <fstream>
#include <vector>
#include <map>
using namespace std;

class ImageInfo{							//abstract class

	static int imageDim;
	static int imageDimensions[3];
	static int frameSize;

public:
	bool SetDim(int d);						//sets image dimensionality (2-3)
	bool SetDimensions(int idx ,int dim);	//indexes 0,1,2
	bool SetFrameSize(int s);

	static int GetDim();							//gets image dimensionality
	static int GetDimensions(int index);			//indexes 0,1,2
	static int GetFrameSize();
};




template <typename imDataType>
class ImageManager : private ImageInfo
{
	static bool singletonFlag;
public:vector<imDataType*> image;						//array of images
	
	ImageManager();

public:
	static ImageManager<imDataType>* Create();
	//bool PrepareBlancImage

	bool Load3D(const char* fname, int frameSize);	//curently with zero-framing
	bool LoadBMP(const char* fname);
	bool SaveBmp(int idx, const char* fname, int slicingDir=-1, int sliPerLine=-1); //defaults: saving 2D image


};

#include "Bacalar/ImageManagerCode.h"
#include "Bacalar/ImageInfo.h"