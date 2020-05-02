/* Authors: Blank, J. and Schmitz, D.
 * This is the main "wrapper" program that computes wave speed
 *
 * Function calls:
 *
 * Compile command: nvcc [FILENAMES HERE] -Xcompiler -O3 -Xcompiler -Wall
 * -Xptxas -O3 -o wrapper
 *
 * FILENAMES: wrapper.cu readLVM.cpp writeCSV.cpp butterHP.cpp butterLP.cpp
 * filtfilt.cpp sort.cpp xcorr.cu
 */

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
  const char *fileIn = "check2.lvm"; // file name to load

  // filenames for checking data
  //const char *filtAcc1File = "filtered_acc1.csv";
  //const char *filtAcc2File = "filtered_acc2.csv";
  //const char *pushPullIndicesFile = "push_pull_indices.csv";

  // file name to which we will write the "push" data
  const char *pushFile = "processed_push.csv";
  /*
const char *releaseFile =
  "processed_release.csv"; // file name to which we will
                           // write the "release" data
                           */
  float **dataMatrix;

  // number of fields (i.e. data columns) in the .lvm file
  int numFields = 0; // (set by readLVM)

  // number of samples (i.e. data rows) in the .lvm file (set by readLVM)
  int numSamples = 0;

  int sampleRate = 50000; // Hz

  // lower and upper cutoff for Butterworth bandpass
  float filter[2] = {150, 5000}; // [Hz]

  // start and end of template window for xcorr
  float window[2] = {0, 2}; // [ms]

  // accelerometer spacing
  float travelDist = 10; // [mm]

  int i; // index variable

  // ----- timing -----
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  // ----- timing -----

  // call readLVM to load data into a matrix
  dataMatrix = readLVM(fileIn, &numFields, &numSamples);

  // filter first accelerometer data
  float *filteredAcc1, *filteredAcc2;
  cudaMallocManaged((void **)&filteredAcc1, sizeof(float) * numSamples);
  cudaMallocManaged((void **)&filteredAcc2, sizeof(float) * numSamples);

  // check that the files were read correctly
  // cout << "Taps: " << dataMatrix[0][2999999] << "\n";
  // cout << "Acc1: " << dataMatrix[1][2999999] << "\n";
  // cout << "Acc2: " << dataMatrix[2][2999999] << "\n";

  filtfilt(dataMatrix[1], filteredAcc1, numSamples, sampleRate, filter[0],
           filter[1]); // filter acc1 data
  filtfilt(dataMatrix[2], filteredAcc2, numSamples, sampleRate, filter[0],
           filter[1]); // filter acc2 data

  // check the filtered signals
  //cout << "Acc1: " << filteredAcc1[99] << "\n";
  //cout << "Acc2: " << filteredAcc2[99] << "\n";
  //writeCSV(filtAcc1File, filteredAcc1, numSamples, 1); // write out push data
  //writeCSV(filtAcc2File, filteredAcc2, numSamples, 1); // write out push data

  /// sort the push and pull tap indices
  size_t *pushPullIndices;
  size_t nLead, nTrail;

  // returns indices of "push" (rising edges) and "release" (falling edges) in
  // array
  // values are ordered as follows: [push1, push2, ... pushN, release1,
  // release2, ... releaseN]
  pushPullIndices =
      sort(dataMatrix[0], sampleRate, numSamples, 100, &nLead, &nTrail);

  // check pushPullIndices
  //writeCSVInt(pushPullIndicesFile, pushPullIndices, (nLead+nTrail), 1);

  // compute wave speed
  bool whichFirst = 0;
  float *push, *release;
  int nPush, nRelease;

  // determine which comes first: push or release. Also determine how many push
  // and release events we can use

  cout << "nLead: " << nLead << "\n";
  cout << "nTrail: " << nTrail << "\n";

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

    // cout << "ind1[i]: " << ind1[i] << "\n";
    // cout << "ind2[i]: " << ind2[i] << "\n";
  } // end for each push

  nPush = nPush - 1;
  nRelease = nRelease - 1;

  computeWaveSpeed(filteredAcc1, filteredAcc2, ind1, ind2, push, nPush, window,
                   sampleRate, travelDist);
  /*
    // RELEASE
    for (i = 0; i < nRelease; i++) {
      if (!whichFirst) {
        ind1[i] = pushPullIndices[nPush + i];
        ind2[i] = pushPullIndices[i + 1];
      } else {
        ind1[i] = pushPullIndices[nPush + i];
        ind2[i] = pushPullIndices[i];
      } // end conditional whichFirst
    }   // end for each release
    computeWaveSpeed(filteredAcc1, filteredAcc2, ind1, ind2, release, nRelease,
                     window, sampleRate, travelDist);
                     */

  // ----- timing -----
  // stop the event and get the elapsed time
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);
  std::cout << "Elapsed Time: " << elapsedTime << "\n";
  // ----- timing -----

  // cout << "push[0]: " << push[0] << "\n";

  writeCSV(pushFile, push, nPush, 1); // write out push data
  // writeCSV(releaseFile, release, nRelease, 1); // write out release data

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