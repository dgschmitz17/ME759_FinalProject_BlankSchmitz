#ifndef FILTER_H
#define FILTER_H

void butterLP(int fc, int fs, float *B, float *A);
void butterHP(int fc, int fs, float *B, float *A);
void filtfilt(float *unfiltered, float *filtered, int N, int fs, float low, float high);

#endif