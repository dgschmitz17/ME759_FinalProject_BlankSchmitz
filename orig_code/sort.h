// Author: Jon Blank

#ifndef SORT
#define SORT

// reads in two arrays of known lengths and computed the cross correlation
int *sort(const double* signalRef, size_t ts, size_t n, size_t tapRate, int* leading, int* trailing);

#endif