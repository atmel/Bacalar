#pragma once
#include "Bacalar/ImageManager.h"


template<typename imDataType>
bool ImageManager<imDataType>::singletonFlag = 0;

template<typename imDataType>
ImageManager<imDataType>* ImageManager<imDataType>::Create(){
	
	if(singletonFlag) return NULL;
	
	return new ImageManager<imDataType>();
}

template<typename imDataType>
ImageManager<imDataType>::ImageManager(){
	
	singletonFlag = 1;
	image.clear();
	gpuImage.clear();
}

/*
!!! READS ONLY 8-BIT UNSIGNED CHAR FILE CURRENTLY !!!

	Load3D loads .hdr/.img image under next free index.
	- if loading multiple images, they must be the same size

	is responsible for initializing all metrics and ImageInfo and
	sending everything regarding image to GPU if UseCuda set

*/
#include <iostream>
template <typename imDataType>
int ImageManager<imDataType>::Load3D(const char* fname, int frameSize){				//without .hdr or .img extension

	ifstream inFile;
	unsigned char rawIn[348];
	int endian = 0;				//1 - little endian, 2 - big endian
	int bitPix;					//bits per pixel

	cout << "Start\n";
	//set image dimension
	if(!SetDim(3)) return BAD_SUBIMAGE_DIMENSION;	//second image has different dimension
	
	//read header file
	string name(fname);
	name.append(".hdr");

	inFile.open(name.c_str());
	if(!inFile.is_open()) return NO_SUCH_FILE;
	inFile.read((char*)rawIn,348);
	inFile.close();

	cout << "File read succesfully\n";
	//check endian
	if((int)rawIn[0] + 256*rawIn[1] == 348) endian = 2;
	if((int)rawIn[4] + 256*rawIn[3] == 348) endian = 1;
	
	bool success = true;

	cout << "endian: " << endian << '\n';

	if(endian == 0){
		//wrong/unknown file format
		return UNKNOWN_FILE_FORMAT;
		//big endian
	}else if(endian == 2){			
		if(/*((int)rawIn[32] + 256*rawIn[33] != 16384) ||*/ (rawIn[38] != 'r'))
			return UNKNOWN_FILE_FORMAT;
		success &= SetDimensions(0,(int)rawIn[42] + 256*rawIn[43]);
		success &= SetDimensions(1,(int)rawIn[44] + 256*rawIn[45]);
		success &= SetDimensions(2,(int)rawIn[46] + 256*rawIn[47]);

		bitPix = rawIn[72];

		//little endian
	}else if(endian == 1){
		if(/*((int)rawIn[35] + 256*rawIn[34] != 16384) ||*/ (rawIn[38] != 'r'))
			return UNKNOWN_FILE_FORMAT;
		success &= SetDimensions(0,(int)rawIn[43] + 256*rawIn[42]);
		success &= SetDimensions(1,(int)rawIn[45] + 256*rawIn[44]);
		success &= SetDimensions(2,(int)rawIn[47] + 256*rawIn[46]);

		bitPix = rawIn[73];
	}
	cout << "Dimensions: " << GetDimensions(0) << ' ' << GetDimensions(1) << ' ' << GetDimensions(2) << '\n';

	if(!success) return BAD_SUBIMAGE_DIMENSIONS;			//second image has different dimension(s)

	if(bitPix != 8) return BAD_IMAGE_FORMAT;				//8-bit unsigned char only

	//read the img file CPU

	if(frameSize != -1){									//save frame size (if first image)
		if(!SetFrameSize(frameSize)) return BAD_FRAMESIZE;
	}

	int curIm = image.size();
	int frame = (frameSize==-1)?GetFrameSize():frameSize;		//first load/subsequent load
	int dims[3] = {GetDimensions(0)+2*frame,	
		GetDimensions(1)+2*frame,GetDimensions(2)+2*frame};		//image dimensions incl. frame

	//name.clear();
	fstream inFile2;
	name = fname;
	name.append(".img");
	
	inFile2.open(name.c_str(),ios::in|ios::binary);
	unsigned char *raw; 

	if(!inFile2.is_open()) return NO_SUCH_FILE;

	ComputeMetrics();				//compute derivated metrics and save them in parrent class	<=====

	cout << "File "<< name << " opened\n";
	PrepareBlankImage(onCPU);		//initialize image vectors

	raw = new unsigned char[GetDimensions(0)*GetDimensions(1)*GetDimensions(2)];
	inFile2.read((char*)raw,GetDimensions(0)*GetDimensions(1)*GetDimensions(2));
	inFile.close();

	unsigned int l=0;

	cout << "Starting reading data\n";

	for(int i=frame; i<dims[2]-frame; i++)
		for(int j=frame;j<dims[1]-frame;j++)
			for(int k=frame;k<dims[0]-frame;k++){		//fastest running,

				if(typeid(imDataType) == typeid(unsigned char)){
					image[curIm][k+(j*dims[0])+(i*dims[1]*dims[0])] = raw[l];
				}else if(typeid(imDataType) == typeid(unsigned int)){
					image[curIm][k+(j*dims[0])+(i*dims[1]*dims[0])] = raw[l]*256;
				}else if(typeid(imDataType) == typeid(float)){
					image[curIm][k+(j*dims[0])+(i*dims[1]*dims[0])] = (unsigned char)(raw[l]/256.0);
				}
				l++;
			}
	//image[curIm][frame+(frame*dims[0])+(frame*dims[1]*dims[0])] = 255; // DEBUG

	cout << "Data read\n";

	if(UseCuda()){				//GPU part
		cout << "Sending image to GPU\n";
		SendToGpu();			//send metrics to gpu
		PrepareBlankImage(onGPU,curIm);
			//send imamge data to GPU
		cudaMemcpy(gpuImage[curIm],image[curIm],sizeof(imDataType)*GetTotalPixelSize(),cudaMemcpyHostToDevice);
		cout << "LoadBMP cuda error:" << cudaGetErrorString(cudaGetLastError()) << '\n';
	}
	
	return true;
}


template <typename imDataType>
int ImageManager<imDataType>::LoadBMP(const char* fname, int frameSize){
	return 0;
}


/*
	Saves image pointed by idx as 2D .bmp, no matter it was 2D or 3D

	Slicing direction is index of dimension, which will transform to the number of slices.
	I.e. 20x30x40 image saved with slicingDir 1 will consist of 30 20x40 planar images

	dim[0] is fastest running
*/
template <typename imDataType>
int ImageManager<imDataType>::SaveBmp(int idx, const char* fname, int slicingDir, int sliPerLine){
	
	if(GetDim() != 3) return false;

	unsigned int bitmapSize[2];			//width, height
	unsigned int i,j, sliceSize, lineSize, blockSize, imWidth, imHeight, imDepth;
	unsigned int frame = GetFrameSize();

	imWidth = GetDimensions(0);
	imHeight = GetDimensions(1);
	imDepth = GetDimensions(2);

	lineSize = (imWidth+2*frame);
	sliceSize = lineSize*(imHeight+2*frame);
	
	switch (slicingDir){
		case 0:
			bitmapSize[1] = (imDepth/sliPerLine + 1)*imHeight;			//height
			bitmapSize[0] = sliPerLine*imWidth;							//width
			blockSize = sliceSize*sliPerLine;
			break;
		case 1:
			bitmapSize[1] = (imHeight/sliPerLine + 1)*imDepth;			//height
			bitmapSize[0] = sliPerLine*imWidth;							//width
			blockSize = lineSize*sliPerLine;
			break;
		case 2:
			bitmapSize[1] = (imWidth/sliPerLine + 1)*imDepth;			//height
			bitmapSize[0] = sliPerLine*imHeight;						//width
			blockSize = sliPerLine;
			break;
	}

	
	//blockSize = sliceSize*sliPerLine;

	//move image from GPU if needed
	if(UseCuda()){
		PrepareBlankImage(onCPU,idx);
		cudaMemcpy(image[idx],gpuImage[idx],sizeof(imDataType)*GetTotalPixelSize(),cudaMemcpyDeviceToHost);
		cout << "SaveBMP cuda error:" << cudaGetErrorString(cudaGetLastError()) << '\n';
	}
	
	unsigned char* imageData = new unsigned char[bitmapSize[0]*bitmapSize[1]];
	imDataType value;

	//fill imageData
	cout << "Starting rearranging data\n";

	for(i=0; i<bitmapSize[0]; i++){			//horizontal
		for(j=0; j<bitmapSize[1]; j++){
			switch(slicingDir){
				case 0:
					value = image[idx][frame*(sliceSize+lineSize+1)+(i%imWidth) +(i/imWidth)*sliceSize 
						+ (j%imHeight)*lineSize + (j/imHeight)*blockSize]; break;
				case 1:
					value = image[idx][frame*(sliceSize+lineSize+1)+(i%imWidth) +(i/imWidth)*lineSize 
						+ (j%imDepth)*sliceSize + (j/imDepth)*blockSize]; break;
				case 2:
					value = image[idx][frame*(sliceSize+lineSize+1)+(i%imHeight)*lineSize +(i/imHeight) 
						+ (j%imDepth)*sliceSize + (j/imDepth)*blockSize]; break;
			}
			if(typeid(imDataType) == typeid(unsigned char)){
				imageData[i+bitmapSize[0]*(bitmapSize[1]-1-j)] = value;
			}else if(typeid(imDataType) == typeid(unsigned int)){
				imageData[i+bitmapSize[0]*(bitmapSize[1]-1-j)] = value/256;
			}else if(typeid(imDataType) == typeid(float)){
				imageData[i+bitmapSize[0]*(bitmapSize[1]-1-j)] = (unsigned char)(value*256.0);
			}
		}
	}
	cout << "Data rearanged\n";

	//save greyscale bmp -- head

	unsigned char *toSave, *palette, uselessChar[16], head[2] = {'B','M'};
	unsigned short int paddingBytes, uniInt;
	unsigned long fileSize, offset, uniLong;
	string name = fname;
	FILE *outFile;

	name.append(".bmp");

	memset(uselessChar,0,16);
	fileSize = 14+40;													//Calculate the file size
	fileSize += 256*4;
	offset = fileSize;
	fileSize += (bitmapSize[0] + 3-((bitmapSize[0]-1)%4))*bitmapSize[1];

	if(!fname) return FILENAME_UNSPECIFIED;								//Open file
	fopen_s(&outFile,name.c_str(),"wb");
	if(!outFile) return UNABLE_TO_OPEN_FILE;				
	
	fwrite(head,2,1,outFile);											//"magic letters" BM
	fwrite(&fileSize,4,1,outFile);										//File size
	fwrite(uselessChar,4,1,outFile);									//reserved
	fwrite(&offset,4,1,outFile);										//Offset (where image data begins)

	uniLong = 40;
	fwrite(&uniLong,4,1,outFile);										//Structure size (2nd header)
	uniLong = bitmapSize[0];
	fwrite(&uniLong,4,1,outFile);										//Width
	uniLong = bitmapSize[1];
	fwrite(&uniLong,4,1,outFile);										//Height
	uniInt = 1;
	fwrite(&uniInt,2,1,outFile);										//Bit-planes
	uniInt = 1*8;
	fwrite(&uniInt,2,1,outFile);										//Bits per pixel
	fwrite(uselessChar,16,1,outFile);									//No compression, stuff
	uniLong = 256;														//Colors in palette
	fwrite(&uniLong,4,1,outFile);	
	fwrite(uselessChar,4,1,outFile);									//Stuff

	//Save indexed greyscale image -- palette, data
	palette = (unsigned char*)malloc(256*4);							//Create color palette
	for(i=0; i<256; i++){
		palette[4*i] = palette[4*i +1] = palette[4*i +2] = i;
		palette[4*i +3] = 0;
	}
	fwrite(palette,1024,1,outFile);										//Save palette
	paddingBytes = 3 - (bitmapSize[0]-1)%4;
	toSave = (unsigned char*)malloc((bitmapSize[0]+paddingBytes)*bitmapSize[1]);		//Prepare image data
	for(i=0; i<bitmapSize[1]; i++){												//(first line last)
		for(j=0; j<bitmapSize[0]; j++){
			toSave[(bitmapSize[1]-1-i)*(bitmapSize[0]+paddingBytes) + j] = imageData[i*bitmapSize[0] + j];
		}																//Padding bytes can be unspecified value
	}
	fwrite(toSave,bitmapSize[1]*(bitmapSize[0]+paddingBytes),1,outFile);
	
	fclose(outFile);
	return true;
}

/*	
	Creates blank image in RAM or in GPU RAM, image is added behind last image
	
	if index is not -1, blank image is created from NULL pointer in image vector<>
	under desired index. If there already is an image, nothing happens

	image vector<> entries are added for both GPU and CPU but one stays NULL

	does not check UseCuda flag
*/
template <typename imDataType>
int ImageManager<imDataType>::PrepareBlankImage(enumWhere wh, int index){
	
	int idx;
	//add new image
	if(index == -1){
		idx = image.size();
		image.push_back(NULL);
		gpuImage.push_back(NULL);
	}else{
		idx = index;
	}

	if(wh == onGPU){
		if(gpuImage[idx] == NULL) cudaMalloc(&gpuImage[idx],sizeof(imDataType)*GetTotalPixelSize());
		cout << "Prepade blank1 cuda error:" << cudaGetErrorString(cudaGetLastError()) << '\n';
		cudaMemset((void*)gpuImage[idx],0,sizeof(imDataType)*GetTotalPixelSize());
		cout << "Prepade blank cuda error:" << cudaGetErrorString(cudaGetLastError()) << '\n';
	}else{
		if(image[idx] == NULL)image[idx] = new imDataType[GetTotalPixelSize()];
		memset(image[idx],0,sizeof(imDataType)*GetTotalPixelSize());
	}
	return true;
}