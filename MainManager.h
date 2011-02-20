#pragma once

/*
	Main definition file - include only this in main

*/

#include "Bacalar/structures.h"
#include "Bacalar/Filter.h"
#include "Bacalar/SEManager.h"
#include "Bacalar/ImageManager.h"
#include "Bacalar/FilterManager.h"


//choose true values for different image types types
//some higher resolution is useless

//allowed image data types:
//unsigned char, unsigned int, float

#define FLOAT_TRUE 1.0
#define CHAR_TRUE 255
#define INT_TRUE 65535

template <typename imDataType> 
class MainManager{
	
	FilterManager<imDataType> *fiM;
	ImageManager<imDataType> *imM;
	SEManager<imDataType> *seM;

	string load, sEls, subF, filter;

public:

	bool LoadFile();
	bool ParseLoadPart();
	bool ParseSEPart();
	bool ParseSubFilterPart();
	bool ParseMainFilterPart();
	bool Launch();

};