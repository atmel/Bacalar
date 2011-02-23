#pragma once

/*

Contains structures and union definition used 
through the program

*/

//choose true values for different image types types
//some higher resolution is useless

//allowed image data types:
//unsigned char, unsigned int, float

#define FLOAT_TRUE (1.0f)
#define CHAR_TRUE (255)
#define INT_TRUE (65535)


//errors
#define BAD_IMAGE_FORMAT (-100)
#define BAD_SUBIMAGE_DIMENSION (-101)
#define NO_SUCH_FILE (-102)
#define UNKNOWN_FILE_FORMAT (-103)
#define BAD_SUBIMAGE_DIMENSIONS (-104)
#define BAD_FRAMESIZE (-105)


