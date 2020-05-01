/*! Authors: Blank, J. and Schmitz, D.
 * Function to write data to a comma-separated file
 */

#include <cstdlib>
#include <iostream>
#include <cstddef>
#include <fstream>

// Provide some namespace shortcuts
using namespace std;

void writeCSV(const char* filename, const float* data, size_t n_rows, size_t n_cols){
    //FILE *fileOut = fopen(filename,"w"); // file name is passed as a function argument, open for write
    ofstream fileOut (filename);
   
    for(size_t i = 0 ; i < n_rows ; i++){
        fileOut << data[i];
        if(i < n_rows){fileOut << "\n";}
    }//end for

    fileOut.close();
}// end WriteCSV