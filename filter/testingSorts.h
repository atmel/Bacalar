#pragma once

/*

	Contains testing (to run on GPU), or outperformed 
	sort/select algorithms

	inluded through fastsort.h

*/


#ifndef TESTSORT
#define TESTSORT

#include "Bacalar/filter/fastsort.h"

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
//------------------------------------------------------------------------------------------------------

template<typename imDataType>
imDataType Filter<imDataType>::Forgetful(imDataType *sortArr, unsigned initBaseLength){

	cout << "new\n";
	unsigned arSize = (initBaseLength/2+2),thread;

	imDataType *_min = sortArr, *_max = sortArr;					//init min,max (indexes to sortArr)
	for(unsigned i=0;i<initBaseLength;i++){
		cout << (unsigned)sortArr[i] << " ";
	}
	cout << "\n";

	for(thread = 1;thread<arSize; thread++){
		//sortArr[thread] = tex1Dfetch(uchar1DTextRef,arrIdx + nb[thread]);
		if(sortArr[thread] > *_max) _max = &(sortArr[thread]);
		if(sortArr[thread] < *_min) _min = &(sortArr[thread]);
	}
			//forgetful sort (loop begins with knowing min, max position)
			//mins are deleted from the [0], maxs from [last] 
	for(unsigned i = 0;;){
			//forget min, max
		cout << "min: " << (unsigned)*_min << " max: " << (unsigned)*_max << '\n';
		if(_min == &sortArr[arSize-1-i]){		//min on max's position
			*_max = sortArr[0];
			cout << "minswap \n";
		}else if(_max == &sortArr[0]){				//max on min's position
			cout << "maxswap\n";
			*_min = sortArr[arSize-1-i];
		}else{										//both condiotions or normal state
			cout << "normal\n";
			*_min = sortArr[0];
			*_max = sortArr[arSize-1-i];
		}
		for(unsigned j=0;i<arSize-j;j++){
			cout << (unsigned)sortArr[j] << " ";
		}
		cout << "\n";

			//end?
		if(initBaseLength%2){			//to spare one/two elements respectively
			if(arSize-i <= 3){		//odd	
				//dst[arrIdx] = sortArr[1];	//position of the median
				return 0;
			}
		}else{
			if(arSize-i <= 4){		//even	
				//dst[arrIdx] = ((unsigned)sortArr[1]+sortArr[2])/2;
				return 0;
			}
		}
			//move new unsorted to [0] -- array shrinks from top
		sortArr[0] = sortArr[arSize+i];
			
			//find new min and max
		_min = sortArr; _max = sortArr;
		i++;
		for(thread = 1;thread<arSize-i; thread++){
			if(sortArr[thread] > *_max) _max = &(sortArr[thread]);
			if(sortArr[thread] < *_min) _min = &(sortArr[thread]);
		}
	}
}


#endif