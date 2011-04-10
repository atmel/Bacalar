#pragma once
#ifndef FILTER_MANAGER
#define FILTER_MANAGER

#include "Bacalar/ImageManager.h"
#include "Bacalar/SEManager.h"
#include "Bacalar/Filter.h"

/*

	

*/

//template <typename imDataType> class FilterManager;
template <typename imDataType> class FilterList;



template <typename imDataType> 
class FilterManager{

	static bool singletonFlag;
	bool useCuda;

	FilterList<imDataType> *mainList;
	vector<FilterList<imDataType>> subList;

	SEManager<imDataType> *SEMan;
	ImageManager<imDataType> *ImagMan;

	FilterManager(SEManager<imDataType> *_SEMan, ImageManager<imDataType> *_ImagMan);

public:

	static FilterManager<imDataType>* Create(SEManager<imDataType> *_SEMan, ImageManager<imDataType> *_ImagMan);

	bool Parse2Filter(string *in);			//parse to mainList
	//bool Parse2CustomFilter(string *in);	//parse/adds to subList
	bool LaunchFilter();
};



template <typename imDataType> 
class FilterList{
public:
	FilterList<imDataType>();

	string name;							//for subFilters
	int length;
	float (Filter<imDataType>::**pf)(imDataType*, int, imDataType*, fourthParam<imDataType>);
	imDataType **dst, **src;
	int *SE;
	fourthParam<imDataType> *par;
};

#include "Bacalar/FilterManagerCode.h"

#endif