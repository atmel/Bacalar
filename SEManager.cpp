#include "SEManager.h"



bool SEManager::Create(int framesize)
{

	int dSize = framesize*framesize*framesize;

	dictionary = new int[3][dsize];

}