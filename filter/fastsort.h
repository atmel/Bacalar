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

#define SWAP(X,Y) {swap=base[(X)];base[(X)]=base[(Y)];base[(Y)]=swap;}


/*

	Simplier version of last function - finds only length/2 element

*/

#define EFFICIENT_MAX 8

template<typename imDataType>
bool Filter<imDataType>::MedianFindOptSimple(imDataType *base, unsigned initBaseLength){
	
	static int first, last;								//we have only pone branch here
	static unsigned length = 0;							//base length (stored for multiple calls)

	if(initBaseLength != 0){							//initialization
		length = initBaseLength;
		return true;
	}
	static unsigned rising, falling, pivot, progress;
	static imDataType swap = 0;								//for SWAP macro

	progress = 2;
	first = 0;								//set full range
	last = length-1;

	while(1){
		if(!progress) return true;
		if(last-first < EFFICIENT_MAX){	//only swap if needed
			//cout << "end\n";
			if(last-(length/2) < (length/2 - 1 - first))
				InsertSortMax(base,first,last,last-(length/2)+2);	
			else
				InsertSortMin(base,first,last,(length/2)-first+1);	
			return true;
		}
		//set pivot as median of first, first+1, last: prevents search to run out of array with no
		//additional condition! (elements must be swapped to the right order)

		pivot = first;
		rising = first+1;
		falling = last;
		if(base[rising] > base[falling]) SWAP(rising,falling);
		if(base[rising] > base[pivot]) SWAP(rising,pivot);
		if(base[falling] < base[pivot]) SWAP(falling,pivot);

		while(1){								//swap all wrong-placed elements
			//fall until first element <= pivot is found
			do falling--; while(base[falling] > base[pivot]);		//thanks to med. no need to check underrun
			//rise until first element >= pivot is found
			do rising++; while(base[rising] < base[pivot]);		//also, no overrun check
			//done and continue or swap and go again
			if(rising > falling){	
				SWAP(pivot,falling);
				break;
			}else{					
				SWAP(rising,falling);
			}
		}
		//find where falling ended and continue with branch containing k-th element
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
}
/*
	Auxiliary insertion sort for small branches
*/
template<typename imDataType>
void Filter<imDataType>::InsertSortMin(imDataType *base, unsigned first, unsigned last, unsigned count){
	static unsigned i,j,min;
	static imDataType swap;

	for(i=0;i<count;i++){				//find only count minimal elements
		min = first+i;
		for(j=min+1; j<=last ;j++){		//but search whole range
			if(base[min] > base[j]) min = j;
		}
		SWAP(min,first+i);
	}
}
template<typename imDataType>
void Filter<imDataType>::InsertSortMax(imDataType *base, unsigned first, unsigned last, unsigned count){
	static unsigned i,j,max;
	static imDataType swap;

	for(i=0;i<count;i++){				//find only count minimal elements
		max = last-i;
		for(j=first;j<last-i;j++){	//but search whole range
			if(base[max] < base[j]) max = j;
		}
		SWAP(max,last-i);
	}
}


/*
Find single k-th element -- used in BES filter
*/
template<typename imDataType>
void Filter<imDataType>::FindKth(imDataType *base, unsigned &first, unsigned &last, unsigned k){

	static unsigned pivot, rising, falling;
	static imDataType swap;			//for SWAP macro

	while(1){
		if(last-first < EFFICIENT_MAX){	//only swap if needed
			//cout << "end\n";
			if(last-k < k-first)
				InsertSortMax(base,first,last,last-k+1);	
			else
				InsertSortMin(base,first,last,k-first+1);	
			return;
		}
		//set pivot as median of first, first+1, last: prevents search to run out of array with no
		//additional condition! (elements must be swapped to the right order)

		pivot = first;
		rising = first+1;
		falling = last;
		if(base[rising] > base[falling]) SWAP(rising,falling);
		if(base[rising] > base[pivot]) SWAP(rising,pivot);
		if(base[falling] < base[pivot]) SWAP(falling,pivot);

		while(1){								//swap all wrong-placed elements
			//fall until first element <= pivot is found
			do falling--; while(base[falling] > base[pivot]);		//thanks to med. no need to check underrun
			//rise until first element >= pivot is found
			do rising++; while(base[rising] < base[pivot]);		//also, no overrun check
			//done and continue or swap and go again
			if(rising > falling){	
				SWAP(pivot,falling);
				break;
			}else{					
				SWAP(rising,falling);
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
		if(!progress) break;
		if(last-first < EFFICIENT_MAX){	//only swap if needed
			//cout << "end\n";
			if(last-(length/2) < (length/2 - 1 - first))
				InsertSortMax(base,first,last,last-(length/2)+2);	
			else
				InsertSortMin(base,first,last,(length/2)-first+1);	
			break;
		}
		//set pivot as median of first, first+1, last: prevents search to run out of array with no
		//additional condition! (elements must be swapped to the right order)

		pivot = first;
		rising = first+1;
		falling = last;
		if(base[rising] > base[falling]) SWAP(rising,falling);
		if(base[rising] > base[pivot]) SWAP(rising,pivot);
		if(base[falling] < base[pivot]) SWAP(falling,pivot);

		while(1){								//swap all wrong-placed elements
			//fall until first element <= pivot is found
			do falling--; while(base[falling] > base[pivot]);		//thanks to med. no need to check underrun
			//rise until first element >= pivot is found
			do rising++; while(base[rising] < base[pivot]);		//also, no overrun check
			//done and continue or swap and go again
			if(rising > falling){	
				SWAP(pivot,falling);
				break;
			}else{					
				SWAP(rising,falling);
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