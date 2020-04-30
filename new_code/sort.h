// Author: Jon Blank

#ifndef SORT
#define SORT

// reads in two arrays of known lengths and computed the cross correlation
size_t *sort(const float* signalRef, size_t ts, size_t n, size_t tapRate, int* leading, int* trailing);

#endif