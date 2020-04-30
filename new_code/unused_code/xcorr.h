// Author: Jon Blank

#ifndef XCORR_H
#define XCORR_H

// reads in two arrays of known lengths and computed the cross correlation
double computeTimeDelay(double *sig1, double *sig2, size_t indA, size_t indZ,
                        double *window, int sampleRate);

void computeWaveSpeed(float *sig1, float *sig2, size_t *indA, size_t *indZ,
                      float *waveSpeed, size_t nInst, float *window,
                      int sampleRate, float travelDist);

#endif