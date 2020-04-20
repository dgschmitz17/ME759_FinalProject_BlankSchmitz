#ifndef FILTER_H
#define FILTER_H

void butterLP(int fc, int fs, double *B, double *A);
void butterHP(int fc, int fs, double *B, double *A);
double *filtfilt(double *unfiltered, int N, int fs, double low, double high);

#endif