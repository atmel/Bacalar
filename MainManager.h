#pragma once

#ifndef MAIN_MANAGER
#define MAIN_MANAGER

/*
	Main definition file - include only this in main

	otherwise --NOT COMPLETE--

	In future should be responsible for lifecycle of program - loading (re)config files, partially parsing them,
	launching filters and optionally communicating with graphic framework

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

#endif