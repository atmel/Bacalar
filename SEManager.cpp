#include "SEManager.h"

template<typename imDataType>
bool SEManager<imDataType>::Create(int framesize)
{

	int dSize = framesize*framesize*framesize;

	dictionary = new int[dsize][3];

}