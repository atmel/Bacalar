#pragma once

#ifndef FASTSORT
#define FASTSORT

#include "Bacalar/Filter.h"
#include <math.h>

/*

	Optimized quicksort:
		- fixed-length stacks (allocated only once per filter function call 
			through optional parameter)

		-

*/

#define SWAP(X,Y) swap=base[(X)];base[(X)]=base[(Y)];base[(Y)]=swap

/*

		stacks need to be signed for extra causes: i.e. base = {1,0,0,0,0.....0}
		will cause last[..] < first[..] and this needs to be captured by the first
		'if(... < 2)' condition!!

*/

template<typename imDataType>
bool Filter<imDataType>::QsortOpt(imDataType *base, unsigned initBaseLength){
	
	static int *first = NULL, *last = NULL;				//stacks (allocated to fixed length - see above)
	static unsigned length = 0;							//base length (stored for multiple calls)

	if(initBaseLength != 0){							//initialization
		if(first != NULL) delete[] first;				//dealocate, if not first call in app
		if(last != NULL) delete[] last;					

		first = new int[initBaseLength];
		last = new int[initBaseLength];
		length = initBaseLength;
		return true;
	}
	static unsigned curStep, rising, falling, pivot;
	static imDataType swap;

	curStep = 0;
	first[0] = 0;										//set full range
	last[0] = length-1;

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

/*

	This function performs partial sorting, resulting in median being in the right place
	It is actually modified quicksort, where branches not containing median are not sorted.
	Sorting ends, when (lengh/2)th and (lengh/2 + 1)th elements are in the right place, so 
	median for odd or even length can be calculated

*/
template<typename imDataType>
imDataType Filter<imDataType>::MedianFindOpt(imDataType *base, unsigned initBaseLength){
	
	static int first, last;								//we have only pone branch here
	static unsigned length = 0;							//base length (stored for multiple calls)

	if(initBaseLength != 0){							//initialization
		length = initBaseLength;
		return true;
	}
	static unsigned progress, rising, falling, pivot;
	static imDataType swap;

	progress = 2;							//is deduced by one, when one of two middle elements is found
	first = 0;								//set full range
	last = length-1;

	while(1){
START:
		if(progress == 0) break;						//both elements are in place
		if(last-first < 2){								//only swap if needed
			if(last-first == 1)
				if(base[first] > base[last]){
					SWAP(first,last);
				}
			progress--;									//els can be so close only at the end
			continue;
		}
		pivot = first;
		rising = first+1;
		falling = last;

		while(1){											//swap all wrong-placed elements
			//fall until first element < pivot is found
			while(base[falling] >= base[pivot]){
				falling--;
				//handle special cases
				if(falling <= pivot){						//everything in range >= pivot -> pivot is in place
					first++;								//only shrink range and run again
					if((falling >= length/2-1)&&(falling <= (length/2))){
						progress--;							//falling points to the pivot, which is in the right place
					}
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
		//find where falling ended and continue with branch containing median
		//falling always contain element, which has been sorted to the right place
		if(falling > (length/2)){			//most trivial cases
			last = falling-1;	
		}else if(falling < (length/2 -1)){	
			first = falling+1;
		}else{ 
			progress--;
			if(falling == length/2-1){
				first = falling+1;
			}else{
				last = falling-1;
			}
		}
	}
	return true;
}

#endif