/*




*/
class ImageInfo{

	static int imageDim;
	static int imageDimensions[3];
	static int frameSize;

public:
	bool SetDim(int d);
	bool SetDimensions(int *dims);
	bool SetFrameSize(int s);

	int GetDim();
	bool GetDimensions(int dims[3]);
	int GetFrameSize();
}




template <class imDataType>
class ImageManager : private ImageInfo
{
imDataType *image;

public:
	bool LoadBMP(const char* fname);
	bool SaveBmp(const char* fname);


}