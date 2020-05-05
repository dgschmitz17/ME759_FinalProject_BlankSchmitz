// GPU Wave Speed Cross-Correlation Code
// Authors: Jonathon Blank and Dylan Schmitz

#include <cstdlib>
#include <cstddef>
#include <cuda.h>
#include <cmath>
#include <iostream>
#include "fileIO.h"

using std::cout;

// Computes the cross-correlation of reference and template, storing the result
// in r.
// Each thread computes the cross-correlation for one tap event.
//
// reference is an array of length ## of managed memory.
// template  is an array of length ## of managed memory.
// r is an array of length ## of managed memory.

__global__ void normxcorr_kernel(float **templ, size_t lenT, float **ref,
                                 size_t lenR, float *r, size_t nInst) {

  // define shorthand for block and thread variables
  int bidx = blockIdx.x;
  int tidx = threadIdx.x;
  int tpb = blockDim.x;
  int nbk = gridDim.x;
  int id = bidx * tpb + tidx;

  size_t nOps = (lenR - lenT + 1);
  size_t rInst = id / nOps;
  size_t rOp = id % nOps;

  size_t n = nInst * nOps;

  /*
  if (id == n - 1) {
    printf("bidx: %d\n", bidx);
    printf("tidx: %d\n", tidx);
    printf("tpb: %d\n", tpb);
    printf("nbk: %d\n", nbk);
    printf("id: %d\n", id);
    printf("nOps: %d\n", nOps);
    printf("rInst: %d\n", rInst);
    printf("rOp: %d\n", rOp);
    printf("n: %d\n", n);
    printf("r[id]: %.1f\n", r[id]);
  } // end if
  */

  float mult, A2, B2;
  if (id >= n) { // conditional for threads that do nothing
    // do nothing
  } else {
    float *my_templ = templ[rInst];
    float *our_ref = ref[rInst];
    float *my_ref = &our_ref[rOp];
    mult = 0;
    A2 = 0;
    B2 = 0;

    for (size_t j = 0; j < lenT; j++) {
      mult += my_templ[j] * my_ref[j];
      A2 += my_templ[j] * my_templ[j];
      B2 += my_ref[j] * my_ref[j];
    } // end for j

    r[id] = mult / sqrt(A2 * B2);

    // if (id == n-1)
    //  printf("r[end]: %f\n", r[id]);

  } // end if..else

} // end xcorr_kernel

// Function computeWaveSpeed
__host__ void computeWaveSpeed(float *sig1, float *sig2, size_t *indA,
                               size_t *indZ, float *waveSpeed, size_t nInst,
                               float *window, int sampleRate,
                               float travelDist) {
  float **templ, **ref;
  cudaMallocManaged((void **)&templ, sizeof(float *) * nInst);
  cudaMallocManaged((void **)&ref, sizeof(float *) * nInst);

  float *rMax = new float[nInst];
  size_t *maxInd = new size_t[nInst];
  float *frameDelay = new float[nInst];
  float *timeDelay = new float[nInst];

  float wo, theta, delta;
  int windowShift, k;
  size_t templLength, temp;
  size_t refLength = 0;

  // determine the beginning index of the template according to the first
  // element of window (which is in milliseconds)
  windowShift = (int)(sampleRate * window[0] / 1000);

  // define the template length according to the second element of window
  templLength = (size_t)(sampleRate * (window[1] - window[0]) / 1000);

  // reset windowShift and templLength if a negative window specified
  if (windowShift < 0){
    windowShift = 0;
    templLength = (size_t)(sampleRate * (window[1]) / 1000);
  }

  // find template and reference for each instance
  for (size_t inst = 0; inst < nInst; inst++) {
    // define the template as a segment of sig1 based on indA and the window
    templ[inst] = &sig1[indA[inst] + windowShift]; 

    // define the reference array length by the difference between indZ and indA
    temp = indZ[inst] - (indA[inst] + windowShift);
    //std::cout << temp << "\n";
    if (inst == 0)
      refLength = temp;
    else
      refLength = (temp < refLength) ? temp : refLength;

    // define the reference as a segment of sig2 starting at the same index as
    // templ
    ref[inst] = &sig2[indA[inst] + windowShift];
    
  } // end for

  // determine shared memory size
  size_t nOps = refLength - templLength + 1;

  // print the number of operations performed
  std::cout << nOps * nInst << "\n";
  std::cout << nInst << "\n";

  // perform the cross-correlation between the template and reference signals
  // float *r = new float[nOps * nInst]; // MANAGED MEMORY
  float *r;
  cudaMallocManaged((void **)&r, sizeof(float) * nOps * nInst);
  size_t threads_per_block =
      1024; // may need to change depending on shared memory size
  size_t number_of_blocks =
      ((nOps * nInst) + threads_per_block - 1) / threads_per_block;

  for (size_t i = 0; i < (nOps * nInst); i++)
    r[i] = 3;

  normxcorr_kernel<<<number_of_blocks, threads_per_block>>>(
      templ, templLength, ref, refLength, r, nInst);

  cudaDeviceSynchronize();

  // const char *rFile = "r_vals.csv";
  // writeCSV(rFile, r, (nOps * nInst), 1); // write out push data

  // find the maximum correlation value
  for (size_t inst = 0; inst < nInst; inst++) {
    rMax[inst] = 0;
    for (size_t i = 0; i < nOps; i++) {
      size_t j = inst * nOps + i;
      if (r[j] > rMax[inst]) {
        rMax[inst] = r[j];
        maxInd[inst] = i;
      } // end if
    }   // end for each element

    // cout << rMax[inst] << "\n";

    k = inst * nOps + maxInd[inst];

    // Performing cosine interpolation to estimate lags with sub-frame
    // precision. See Cespedes et al., Ultrason Imaging 17, 142-171 (1995).
    if (maxInd[inst] > 0 && maxInd[inst] < (refLength - templLength)) {
      wo = acos((r[k - 1] + r[k + 1]) / (2 * r[k]));
      theta = atan((r[k - 1] - r[k + 1]) / (2 * r[k] * sin(wo)));
      delta = -theta / wo;
      frameDelay[inst] = maxInd[inst] - 1 + delta;
      //cout << inst << " delta is " << delta << "\n";
    } else {
      frameDelay[inst] = maxInd[inst] - 1;
    } // end if/else

    // compute time lag based on frame lag in ms
    timeDelay[inst] = (((float)frameDelay[inst]) / sampleRate) * 1000;

    waveSpeed[inst] = travelDist / timeDelay[inst];
  } // end for each instance

  // clean up
  cudaFree(r);
  cudaFree(ref);
  cudaFree(templ);

  delete[] rMax;
  delete[] maxInd;
  delete[] frameDelay;
  delete[] timeDelay;

} // end computeTimeDelay

/*
size_t nInstPerBlock =
  (threads_per_block / nOps) + (((threads_per_block % nOps) > 1) ? 2 : 1);
size_t shared_memory_size =
  nInstPerBlock * (sizeof(float) * (refLength + templLength));

  normxcorr_kernel<<<number_of_blocks, threads_per_block, shared_memory_size>>>(
      templ, templLength, ref, refLength, r, nInst);
      */

// split up the shared memory with pointers to the appropriate elements
/*
extern __shared__ float sm[];
float *image_shared = sm;
float *mask_shared = (float *)&image_shared[tpb + 2 * R];
float *output_shared = (float *)&mask_shared[2 * R + 1];

if (id >= n) { // conditional for threads that do nothing
  // do nothing
} else {
  // load elements of image into image_shared
  if (t == 0) { // first thread gets preceding elements
    for (int u = 0; u <= R; u++) {
      // make sure we aren't accessing image out-of-bounds
      if ((id - (int)R + (int)u) >= 0) {
        image_shared[u] = image[id - R + u];
      } else {
        image_shared[u] = 0.0; // zero padding at beginning
      }
    }
  } else if (t == (tpb - 1) ||
             id == (n - 1)) { // last thread gets succeeding elements
    for (int u = 0; u <= R; u++) {
      // make sure we aren't accessing image out-of-bounds
      if ((id + (int)u) < n) {
        image_shared[tpb + R - 1 + u] = image[id + u];
      } else {
        image_shared[tpb + R - 1 + u] = 0.0; // zero padding at end
      }
    }
  } else { // all other threads just fetch their corresponding element
    image_shared[R + t] = image[id];
  } // end if..else if..else

  // load elements of mask into mask_shared
  if (t < (2 * R + 1)) {
    mask_shared[t] = mask[t];
  } // end if
}   // end if..else

__syncthreads(); // syncthreads out of branch to prevent hang
*/