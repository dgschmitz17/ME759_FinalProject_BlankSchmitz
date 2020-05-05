/* Authors: Blank, J. and Schmitz, D.
 * This is the main "wrapper" program that computes wave speed
 *
 * Compile command: nvcc [FILENAMES HERE] -Xcompiler -O3 -Xcompiler -Wall
 * -Xptxas -O3 -o wrapper
 *
 * FILENAMES: wrapper.cu readCSV.cpp writeCSV.cpp butterHP.cpp butterLP.cpp
 * filtfilt.cpp sort.cpp xcorr.cu
 */

 // DRIVE FOLDER CONTAINING DATA FOR PROCESSING
 // https://drive.google.com/drive/folders/1O8VQ0D-_w_4_VmT_K8sK0B8mNpU4eNX1?usp=sharing

// ----- LIBRARIES ----- //
#include <cstdlib>
#include <cstdbool>
#include <iostream>
#include <cstddef>
#include <cuda.h>

#include "fileIO.h"
#include "filter.h"
#include "sort.h"
#include "xcorr.cuh"

// provide namespace shortcuts
using std::cout;

// ----- BEGIN MAIN PROGRAM -----
int main(int argc, char *argv[]) {
  const char *fileIn = "data.csv"; // file name to load

  // file name to which we will write the "push" data
  const char *pushFile = "processed_push.csv";

  float **dataMatrix; // raw data stored here after read

  // number of fields (i.e. data columns) in the .lvm file
  int numFields = 0; // (set by readLVM)

  // number of samples (i.e. data rows) in the .lvm file (set by readLVM)
  int numSamples = 0;

  int sampleRate = 51200; // Hz

  // lower and upper cutoff for Butterworth bandpass
  float filter[2] = {150, 5000}; // [Hz]

  // start and end of template window for xcorr
  float window[2] = {-2, 2}; // [ms]

  // accelerometer spacing
  float travelDist = 8; // [mm]

  int i; // index variable

  // ----- timing -----
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  // ----- timing -----

  // call readLVM to load data into a matrix
  dataMatrix = readCSV(fileIn, &numFields, &numSamples);

  // filter first accelerometer data
  float *filteredAcc1, *filteredAcc2;
  cudaMallocManaged((void **)&filteredAcc1, sizeof(float) * numSamples);
  cudaMallocManaged((void **)&filteredAcc2, sizeof(float) * numSamples);

  //filter each acc signal independently
  filtfilt(dataMatrix[0], filteredAcc1, numSamples, sampleRate, filter[0],
           filter[1]); // filter acc1 data
  filtfilt(dataMatrix[1], filteredAcc2, numSamples, sampleRate, filter[0],
           filter[1]); // filter acc2 data

  /// sort the push and pull tap indices
  size_t *pushPullIndices;
  size_t nLead, nTrail;

  // returns indices of "push" (rising edges) and "release" (falling edges) in
  // array
  // values are ordered as follows: [push1, push2, ... pushN, release1,
  // release2, ... releaseN]
  pushPullIndices =
      sort(dataMatrix[2], sampleRate, numSamples, 100, &nLead, &nTrail);

  // compute wave speed
  bool whichFirst = 0;
  float *push, *release;
  int nPush, nRelease;

  // determine which comes first: push or release. Also determine how many push
  // and release events we can use

  // ------------------ PUSH ............ PUSH ------------------
  if (nLead > nTrail) { // push starts and ends, so throw away last push index,
                        // both push and release have length nTrail
    nPush = nTrail;
    nRelease = nTrail;

    // ------------------ RELEASE ............ RELEASE ------------------
  } else if (nLead < nTrail) { // release starts and ends, so throw away last
                               // release index, both push and release have
                               // length nLead
    whichFirst = 1;
    nPush = nLead;
    nRelease = nLead;

    // ------------------ PUSH ............ RELEASE ------------------
    // ------------------ RELEASE ............ PUSH ------------------
  } else { // same number of push and release, so need to determine which comes
           // first and how many of each
    // push first = 0, release first = 1
    if (pushPullIndices[0] >
        pushPullIndices[nLead]) { // determine which is first
      whichFirst = 1;
    } // end determine whichFirst

    if (!whichFirst) { // push is first, push will have one more event than
                       // release
      nPush = nLead;
      nRelease = nLead - 1;

    } else { // release is first, release will have one more event than push
      nPush = nTrail - 1;
      nRelease = nTrail;
    }
  } // if malloc based on relative push/release length

  push = new float[nPush];
  release = new float[nRelease];

  // set index arrays to larger needed size for push/release
  size_t nInds = (nRelease < nPush) ? nPush : nRelease;
  size_t *ind1 = new size_t[nInds];
  size_t *ind2 = new size_t[nInds];

  // ------------------ COMPUTE WAVE SPEED ------------------
  // PUSH
  // determine indices of each instance
  for (i = 0; i < nPush; i++) {
    // determine event indices to pass to computeTimeDelay
    if (!whichFirst) {
      ind1[i] = pushPullIndices[i];
      ind2[i] = pushPullIndices[nPush + i];
    } else {
      ind1[i] = pushPullIndices[i];
      ind2[i] = pushPullIndices[nPush + i + 1];
    } // end conditional whichFirst
  } // end for each push

  // adjust to prevent out-of-bounds access at end
  nPush = nPush - 1;
  nRelease = nRelease - 1;

  //call function to compute wave speed. contains kernel.
  computeWaveSpeed(filteredAcc1, filteredAcc2, &ind1[1], &ind2[1], push, nPush-1, window,
                   sampleRate, travelDist);

  // ----- timing -----
  // stop the event and get the elapsed time
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);
  std::cout << "Elapsed Time: " << elapsedTime << "\n";
  // ----- timing -----

  writeCSV(pushFile, push, nPush, 1); // write out push data

  // free all allocated memory
  delete[] dataMatrix;
  delete[] pushPullIndices;
  delete[] push;
  delete[] release;
  delete[] ind1;
  delete[] ind2;

  cudaFree(filteredAcc1);
  cudaFree(filteredAcc2);

  return 0;
} // end main