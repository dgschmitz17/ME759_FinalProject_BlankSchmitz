// Author: Jon Blank

#ifndef XCORR_CUH
#define XCORR_CUH

void computeWaveSpeed(float *sig1, float *sig2, size_t *indA, size_t *indZ,
                      float *waveSpeed, size_t nInst, float *window,
                      int sampleRate, float travelDist);

#endif