#pragma once
#include "Bacalar/FilterManager.h"

template <typename imDataType> 
FilterList<imDataType>::FilterList(){
	
	lenght = 0;
	name.clear();
}


//--------------------------------FilterManager
template<typename imDataType>
bool FilterManager<imDataType>::singletonFlag = 0;


template <typename imDataType> 
FilterManager<imDataType>* FilterManager<imDataType>::Create(SEManager<imDataType> *_SEMan, ImageManager<imDataType> *_ImagMan){

	if(singletonFlag) return NULL;

	return new FilterManager<imDataType>(_SEMan,_ImagMan);
}

template <typename imDataType> 
FilterManager<imDataType>::FilterManager(SEManager<imDataType> *_SEMan, ImageManager<imDataType> *_ImagMan){
	
	SEMan = _SEMan;
	ImagMan = _ImagMan;

	mainList = new FilterList<imDataType>();		//create empty list
	subList.clear();
}




template <typename imDataType>
bool FilterManager<imDataType>::Parse2Filter(string *in){
	return 1; 
}

////template <typename imDataType> 
////bool FilterManager<imDataType>::Parse2CustomFilter(string *in){
////
////}

template <typename imDataType> 
bool FilterManager<imDataType>::LaunchFilter(){
	return 1; 
}