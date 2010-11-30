#include "ImageManager.h"

/*#define UCH unsigned char
typedef struct _hdr{
	UCH sizeof_hdr[4];			//should be 348, also check for endian type
	UCH text[28];
	UCH extents[4];				//should be 16384 - image minimum extent size
	UCH un1[2];
	UCH regular;				//must be 'r'
	UCH un2;
	UCH dim1[2];
	UCH dim2[2];
	UCH dim3[2];
	UCH dimt[2];				//volumes in file
	//uninteresting data
} hdr;

typedef union _tanshdr{
	UCH raw[348];
	hdr header;
}
#undef UCH*/

template <typename imDataType>
bool ImageManager<imDataType>::Load3D(const char* fname){				//without .hdr or .img extension

	ifstream inFile;
	unsigned char rawIn[348];
	int endian = 0;				//1 - little endian, 2 - big endian

	//set image dimension
	SetDim(3);
	
	//read header file
	string name(fname);
	name.append(".hdr");

	inFile.open(name.c_str());
	inFile.read(rawIn,348);
	inFile.close();

	//check endian
	if((int)rawIn[0] + 256*rawIn[1] == 348) endian = 2;
	if((int)rawIn[4] + 256*rawIn[3] == 348) endian = 1;
	
	if(endian == 0){
		//wrong/unknown file format
		return 0;
		//big endian
	}else if(endian == 2){			
		if(((int)rawIn[32] + 256*rawIn[33] != 16384) || (rawIn[38] != 'r'))
			return 0;
		SetDimensions(1,(int)rawIn[42] + 256*rawIn[43]);
		SetDimensions(2,(int)rawIn[44] + 256*rawIn[45]);
		SetDimensions(3,(int)rawIn[46] + 256*rawIn[47]);


		//little endian
	}else if(endian == 1){
		if(((int)rawIn[35] + 256*rawIn[34] != 16384) || (rawIn[38] != 'r'))
			return 0;
	}
}


template <typename imDataType>
bool ImageManager<imDataType>::LoadBMP(const char* fname){

}


template <typename imDataType>
bool ImageManager<imDataType>::SaveBmp(const char* fname, int slicingDir=-1, int sliPerLine=-1){

}