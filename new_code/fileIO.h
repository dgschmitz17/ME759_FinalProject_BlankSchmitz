// Author: Dylan Schmitz

#ifndef READ_LVM
#define READ_LVM

#include <cstdlib>

float ** readLVM(const char *filename,int *numRows,int *numCols);
//double * readLVM(char *filename,int *numRows,int *numCols);

void writeCSV(const char* filename, const float* data, size_t n_rows, size_t n_cols);

void writeCSVInt(const char* filename, const int* data, size_t n_rows, size_t n_cols);

#endif