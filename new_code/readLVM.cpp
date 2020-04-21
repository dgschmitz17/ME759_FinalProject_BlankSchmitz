/*! Authors: Blank, J. and Schmitz, D.
 * This is the function that reads values from an .lvm file
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <cstddef>

// Provide some namespace shortcuts
using std::cout;

double ** readLVM(char *filename,int *numCols,int *numRows){
//int main(void){
    //FILE *fileIn  = fopen("check1.lvm", "r"); // read only
    FILE *fileIn = fopen(filename,"r"); // file name is passed as a function argument, open read only
    int len = 100; // maximum buffer length for fgets
    char buf[100]; // declare buffer as a character array

    int numHeadRows = 23; // .lvm files have 23 lines of non-data text at the top
    int numEofReturns = 1; // .lvm files have an extra return after the last row of data
    int numFileRows = 0; // initialize number of rows in file to zero before counting
    int numDataRows; // computed from numFileRows, numHeaderRows, and numEofReturns
    int numDataCols = 0; // found by parsing the first line of data
    int rowIndex = 0;
    int colIndex;

    char *ptr; // pointer used by strtok to split the data buffer
    char *tok = "\t"; // .lvm files are tab delimited

    int i; // index variable
           
    // make sure the file exists
    if ( fileIn == NULL ) {
        cout << "Error! Could not open file." << "\n"; 
        exit(-1);
    }// end if fileIn doesn't exist 

    // count the number of rows and columns in the file
    while( fgets(buf,len,fileIn) ){
        numFileRows++; // count number of rows in the file

        // use strtok to count the number of data columns
        if( numFileRows == 24){
            ptr = strtok(buf, tok); // begin tokenizing the buffer
	        while(ptr != NULL){ // as long as the pointer doesn't hit the end of the buffer
                numDataCols++;
		        ptr = strtok(NULL, tok); // move the pointer to the next token in the buffer
	        }// end while
        }// end if to count columns
    }//end while fgets doesn't hit EOF
    numDataRows = numFileRows - numHeadRows - numEofReturns; // compute the number of data rows
    *numRows = numDataRows; // pass back the number of rows (samples)
    *numCols = numDataCols; // pass back the number of columns (fields)
    rewind(fileIn); // reset file pointer to the beginning of the file

    // allocate memory for output matrix
    double **matrixOut = (double **) malloc(sizeof(double *) * numDataCols);

    for( i = 0 ; i < numDataCols ; i++ ){
        matrixOut[i] = (double *) malloc(sizeof(double) * numDataRows);
    }// end for columns to allocate memory

    // read the file again, but this time parse each column into its own array
    for( i = 0 ; i < (numFileRows - numEofReturns) ; i++ ){
        fgets(buf, len, fileIn);
        if( i >= numHeadRows){ // parsing only happens on data rows
            // parse data row into fields based on columns, tab separated
	        ptr = strtok(buf, tok); // begin tokenizing the buffer

            colIndex = 0;
	        while(ptr != NULL){ // as long as the pointer doesn't hit the end of the buffer
                matrixOut[colIndex % numDataCols][rowIndex] = atof(ptr);
		        ptr = strtok(NULL, tok); // move the pointer to the next token in the buffer
                colIndex++;
	        }// end while

            rowIndex++;
        }// end if index is for a data row
    }//end while i is less than numFileRows

    return matrixOut;
}// end readCSV