/*! Authors: Blank, J. and Schmitz, D.
 * This is the function that reads values from an .lvm file
 */

#include <cstdlib>
#include <string>
#include <iostream>
#include <cstddef>
#include <fstream>
#include <string>
#include <bits/stdc++.h>

// Provide some namespace shortcuts
using namespace std;
using std::strcpy;
using std::cout;

float **readCSV(const char *filename, int *numCols, int *numRows) {
  string line;               // string for getline()
  char buf[100];             // buffer for tokenizing
  ifstream fileIn(filename); // open input file

  // make sure the file exists
  if (!fileIn.is_open()) {
    cout << "Error! Could not open file."
         << "\n";
    exit(-1);
  } // end if fileIn doesn't exist

  // .lvm files have 23 lines of non-data text at the top
  int numHeadRows = 0;

  // .lvm files have an extra return after the last row of data
  int numEofReturns = 0;

  // initialize number of rows in file to zero before counting
  int numFileRows = 0;

  // computed from numFileRows, numHeaderRows, and numEofReturns
  int numDataRows;

  // found by parsing the first line of data
  int numDataCols = 0;
  int rowIndex = 0;
  int colIndex;

  char *ptr;              // pointer used by strtok to split the data buffer
  const char *tok = ","; // .lvm files are tab delimited

  int i; // index variable

  // count the number of rows and columns in the file
  while (getline(fileIn, line)) {
    numFileRows++; // count number of rows in the file
    strcpy(buf, line.c_str());

    // use strtok to count the number of data columns
    if (numFileRows == 1) {
      ptr = strtok(buf, tok); // begin tokenizing the buffer

      // as long as the pointer doesn't hit the end of the buffer
      while (ptr != NULL) {
        numDataCols++;
        // move the pointer to the next token in the buffer
        ptr = strtok(NULL, tok);
      } // end while
    }   // end if to count columns
  }     // end while fgets doesn't hit EOF

  // compute the number of data rows
  numDataRows = numFileRows - numHeadRows - numEofReturns;
  *numRows = numDataRows; // pass back the number of rows (samples)
  *numCols = numDataCols; // pass back the number of columns (fields)

  // reset file pointer to the beginning of the file
  fileIn.clear();
  fileIn.seekg(0);

  // allocate memory for output matrix
  float **matrixOut = new float *[numDataCols];

  for (i = 0; i < numDataCols; i++) {
    matrixOut[i] = new float[numDataRows];
  } // end for columns to allocate memory

  // read the file again, but this time parse each column into its own array
  for (i = 0; i < (numFileRows - numEofReturns); i++) {
    getline(fileIn, line);
    strcpy(buf, line.c_str());

    if (i >= numHeadRows) { // parsing only happens on data rows
      // parse data row into fields based on columns, tab separated
      ptr = strtok(buf, tok); // begin tokenizing the buffer

      colIndex = 0;
      // as long as the pointer doesn't hit the end of the buffer
      while (ptr != NULL) {
        matrixOut[colIndex % numDataCols][rowIndex] = atof(ptr);

        // move the pointer to the next token in the buffer
        ptr = strtok(NULL, tok);
        colIndex++;
      } // end while

      rowIndex++;
    } // end if index is for a data row
  }   // end while i is less than numFileRows

  fileIn.close();
  return matrixOut;
} // end readCSV