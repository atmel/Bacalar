#pragma once

/*
	Main definition file - include only this in main

*/

#include "Bacalar/structures.h"
#include "Bacalar/Filter.h"
#include "Bacalar/SEManager.h"
#include "Bacalar/ImageManager.h"
#include "Bacalar/FilterManager.h"


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