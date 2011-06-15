#pragma once

#ifndef FASTSORT
#define FASTSORT

#include "Bacalar/Filter.h"
#include <math.h>

/*

	Optimized quicksort:
		- fixed-length stacks (allocated only once per filter function call 
			through optional parameter)

*/

#define SWAP(X,Y) swap=base[(X)];base[(X)]=base[(Y)];base[(Y)]=swap


/*

	This function performs partial sorting, resulting in median being in the right place
	It is actually modified quicksort, where branches not containing median are not sorted.
	Sorting ends when (lengh/2)th and (lengh/2 + 1)th elements are in the right place, so 
	median for odd or even length can be calculated

*/
template<typename imDataType>
bool Filter<imDataType>::MedianFindOpt(imDataType *base, unsigned initBaseLength){
	
	static int first, last;								//we have only pone branch here
	static unsigned length = 0;							//base length (stored for multiple calls)

	if(initBaseLength != 0){							//initialization
		length = initBaseLength;
		return true;
	}
	static unsigned progress, rising, falling, pivot;
	static imDataType swap;								//for SWAP macro

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

/*
Find single k-th element -- used in BES filter
*/
template<typename imDataType>
void Filter<imDataType>::FindKth(imDataType *base, unsigned &first, unsigned &last, unsigned k){

	static unsigned pivot, rising, falling;
	static imDataType swap;			//for SWAP macro

	while(1){
START_KTH:
		if(last-first < 2){								//only swap if needed
			if(last-first == 1)
				if(base[first] > base[last]){
					SWAP(first,last);
				}				
			return;
		}
		pivot = first;
		rising = first+1;
		falling = last;

		while(1){										//swap all wrong-placed elements
			//fall until first element < pivot is found
			while(base[falling] >= base[pivot]){
				falling--;
				//handle special cases
				if(falling <= pivot){					//everything in range >= pivot -> pivot is in place
					first++;							//only shrink range and run again
					if(falling == k) return;			//falling points to the pivot, which is in the right place, DONE
					goto START_KTH;							//escape from multiple cycles, save one variable
				}
			}
			//rise until first element >= pivot is found
			while((base[rising] < base[pivot])&&(rising < falling)){
				rising++;
			}
			//done and continue or swap and go again
			if(rising != falling){						//swap rising and falling
				SWAP(rising,falling);
			}else{										//swap pivot and this
				SWAP(pivot,falling);
				break;
			}
		}
		//find where falling ended and continue with branch containing k-th element
		//falling always contain element, which has been sorted to the right place
		if(falling > k){			//most trivial cases
			last = falling-1;	
		}else if(falling < k){	
			first = falling+1;
		}else{ 
			return;
		}
	}
}

/*
	Uses modified quickselect to ensure that all inputs for BES/WBES ar in the right place.
	First it finds median in the same way as in FindMedianOpt but it keeps track of elements 
	that are in right position and when searching for 1st and 3rd quartil, it searches only 
	between two closest sorted elemets (two for each quartil)
*/

template<typename imDataType>
bool Filter<imDataType>::UniBESFind(imDataType *base, unsigned initBaseLength){
	
	static int first, last;								//we have only pone branch here
	static unsigned length = 0, q1pos, q3pos;			//base length (stored for multiple calls)

	if(initBaseLength != 0){							//initialization
		length = initBaseLength;
		q1pos = ceil((float)length/4)-1;				//BES constatants
		q3pos = floor((float)(3*length+4)/4)-1;
		return true;
	}
	static unsigned progress, rising, falling, pivot;
	static imDataType swap;								//for SWAP macro
	static unsigned q1min, q1max, q3min, q3max;

	progress = 2;							//is deduced by one, when one of two middle elements is found
	q1min = first = 0;						//set full range
	q3max = last = length-1;
	q1max = q3min = length/2;

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
		//falling still points to the element whis is in the right place: constrain search for 1,3 quartil

		if(falling < q3pos){
			if(falling > q1pos){
				if(q1max > falling) q1max = falling;
				if(q3min < falling) q3min = falling;
			}else if(falling < q1pos){
				if(q1min < falling) q1min = falling;
				if(q3min < falling) q3min = falling;
			}else{
				q1min = q1pos;				//to prevent search of q1
			}
		}else if(falling > q3pos){
			if(q1max > falling) q1max = falling;
			if(q3max > falling) q3max = falling;
		}else{
			q3min = q3pos;					//to prevent search of q3
		}
	}
	//now find q1 and q3
	if(q1min != q1pos) FindKth(base,q1min,q1max,q1pos);
	if(q3min != q3pos) FindKth(base,q3min,q3max,q3pos);

	return true;
}

#include "Bacalar/filter/testingSorts.h"

#endif