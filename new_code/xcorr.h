// Author: Jon Blank

#ifndef XCORR_H
#define XCORR_H

// reads in two arrays of known lengths and computed the cross correlation
double computeTimeDelay(double *sig1, double *sig2, size_t indA, size_t indZ, double *window, int sampleRate);

#endif