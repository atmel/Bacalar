#pragma once

#include "Bacalar/cuda/CudaInfo.h"
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

	static unsigned imageDim, imageDimensions[3], frameSize;

	//image metrics
	static unsigned imageLineSize, imageSliceSize, imageTotalPixelSize;

public:
	bool SetDim(int d);						//sets image dimensionality (2-3)
	bool SetDimensions(int idx ,int dim);	//indexes 0,1,2
	bool SetFrameSize(int s);
	bool ComputeMetrics();
	bool SendToGpu();

		//static because of filter is abstract and need to use it
	static inline unsigned GetDim(){return imageDim;}					
	static inline unsigned GetDimensions(unsigned index){return imageDimensions[index];}
	static inline unsigned GetFrameSize(){return frameSize;}
	static inline unsigned GetLineSize(){return imageLineSize;}
	static inline unsigned GetSliceSize(){return imageSliceSize;}
	static inline unsigned GetTotalPixelSize(){return imageTotalPixelSize;}
};




template <typename imDataType>
class ImageManager : private ImageInfo, private CudaInfo
{
	static bool singletonFlag;
public:vector<imDataType*> image;						//array of images
	   vector<imDataType*> gpuImage;
	
	ImageManager();

public:
	static ImageManager<imDataType>* Create();
	int PrepareBlankImage(bool gpu, int index =-1);					//creates empty image for filter result

	int Load3D(const char* fname, int frameSize = -1);	//curently with zero-framing
	int LoadBMP(const char* fname, int frameSize = -1);
	int SaveBmp(int idx, const char* fname, int slicingDir=-1, int sliPerLine=-1); //defaults: saving 2D image


};

#include "Bacalar/ImageManagerCode.h"
#include "Bacalar/ImageInfo.h"