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

	This function performs partial sorting, resulting in median being in the right place
	It is actually modified quicksort, where branches not containing median are not sorted.
	Sorting ends when (lengh/2)th and (lengh/2 + 1)th elements are in the right place, so 
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

#include "Bacalar/filter/testingSorts.h"

#endif