// Author: Dylan Schmitz

#ifndef READ_CSV
#define READ_CSV

#include <cstdlib>

//float ** readLVM(const char *filename,int *numRows,int *numCols);

float ** readCSV(const char *filename,int *numRows,int *numCols);

void writeCSV(const char* filename, const float* data, size_t n_rows, size_t n_cols);

void writeCSVInt(const char* filename, const size_t* data, size_t n_rows, size_t n_cols);

#endif