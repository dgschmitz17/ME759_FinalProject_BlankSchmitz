// Author: Dylan Schmitz

#ifndef READ_LVM
#define READ_LVM

#include <stdlib.h>

double ** readLVM(char *filename,int *numRows,int *numCols);
//double * readLVM(char *filename,int *numRows,int *numCols);

void writeCSV(const char* filename, const double* data, size_t n_rows, size_t n_cols);

void writeCSVInt(const char* filename, const int* data, size_t n_rows, size_t n_cols);

#endif