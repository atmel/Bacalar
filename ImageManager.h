#pragma once

/*




*/

#include <fstream>
using namespace std;

class ImageInfo{

	static int imageDim;
	static int imageDimensions[3];
	static int frameSize;

public:
	bool SetDim(int d);
	bool SetDimensions(int *dims);
	bool SetFrameSize(int s);

	int GetDim();
	bool GetDimensions(int index, int dim);
	int GetFrameSize();
};




template <typename imDataType>
class ImageManager : private ImageInfo
{
imDataType **image;						//array of images

public:
	bool Load3D(const char* fname);
	bool LoadBMP(const char* fname);
	bool SaveBmp(const char* fname, int slicingDir=-1, int sliPerLine=-1); //defaults: saving 2D image


};