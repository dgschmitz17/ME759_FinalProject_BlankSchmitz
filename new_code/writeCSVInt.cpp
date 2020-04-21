// Function to write data to a comma-separated file
// Author: Dylan Schmitz

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <cstddef>

// Provide some namespace shortcuts
using std::cout;

void writeCSVInt(const char* filename, const int* data, size_t n_rows, size_t n_cols){
    FILE *fileOut = fopen(filename,"w"); // file name is passed as a function argument, open for write
    int i, j; // loop index variables

    // loop through the input array, splitting into rows and columns according to the input sizes
    /*
    for(i = 0 ; i <= n_rows ; i++){
        for(j = 0 ; j < n_cols ; j++){
            fprintf(fileOut,"%.12f",data[j][i]);
            if(j != n_cols-1){fprintf(fileOut,",");}
        }// end for cols
        fputs("\n",fileOut);
    } // end for rows
    */
   
    for(i = 0 ; i < n_rows ; i++){
        cout << fileOut;
        cout << data[i] << "\n";
        if(i < n_rows){fputs("\n",fileOut);}
    }//end for

    fclose(fileOut);
}// end WriteCSV