#pragma once

#ifndef FASTSORT
#define FASTSORT

#include "Bacalar/Filter.h"
#include <math.h>

/*

	Optimized quicksort:
		- fixed-lenght stacks (allocated only once per filter function call 
			through optional parameter)

		-

*/

#define SWAP(X,Y) swap=base[(X)];base[(X)]=base[(Y)];base[(Y)]=swap


template<typename imDataType>
bool Filter<imDataType>::QsortOpt(imDataType *base, unsigned initBaseLenght){
	
	static imDataType *first = NULL, *last = NULL;		//stacks (allocated to fixed lenght - see above)
	static unsigned lenght = 0;							//base lenght (stored for multiple calls)

	if(initBaseLenght != 0){							//initialization
		if(first != NULL) delete[] first;				//dealocate, if not first call in app
		if(last != NULL) delete[] last;					
		//unsigned len = log10((float)initBaseLenght)/log10(2.0f) + 10;
		first = new imDataType[initBaseLenght/2];
		last = new imDataType[initBaseLenght/2];
		lenght = initBaseLenght;
		return true;
	}

	static unsigned curStep, rising, falling, pivot;
	static imDataType swap;

	curStep = 0;
	first[0] = 0;										//set full range
	last[0] = lenght-1;

	while(1){
START:
		if(last[curStep]-first[curStep] < 2){			//too short range
			if(last[curStep]-first[curStep] == 1){		//only swap if needed
				if(base[first[curStep]] > base[last[curStep]]){
					SWAP(first[curStep],last[curStep]);
				}
			}
			//step back to previous level or end
			if(curStep == 0) break;						//this is the only end
			curStep--;
			continue;									//we cannot be sure if previous is not also too short
		}
		pivot = first[curStep];
		rising = first[curStep]+1;
		falling = last[curStep];

		while(1){											//swap all wrong-placed elements
			//fall until first element < pivot is found
			while(base[falling] >= base[pivot]){
				falling--;
				//handle special cases
				if(falling <= pivot){						//everything in range >= pivot -> pivot is in place
					first[curStep]++;						//only shrink range and run again
					goto START;								//escape from multiple cycles, save one variable
				}
			}
			//rise until first element >= pivot is found
			while((base[rising] < base[pivot])&&(rising < falling)){
				rising++;
			}
			//done and continue or swap and go again
			if(rising != falling){							//swap rising and falling
				SWAP(rising,falling);
			}else{											//swap pivot and this
				SWAP(pivot,falling);
				break;
			}
		}
		//split into 2 branches (virtually through using stack and incrementing curStep)
		//pair deeper in the stack is the branch with smaller indexes
		//first[curStep] -- let as it is
		last[curStep+1] = last[curStep];
		last[curStep] = falling-1;
		curStep++;
		first[curStep] = falling+1;
	}
	return true;
}

#endif