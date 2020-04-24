/* Authors: Blank, J. and Schmitz, D.
 * This is the main "wrapper" program that computes wave speed
 *
 * Function calls:
 *
 * Compile command: COMPILE COMMAND HERE
 */

// ----- LIBRARIES ----- //
#include <cstdlib>
#include <cstdbool>
#include <iostream>
#include <chrono>
#include <ratio>
#include <cmath>
#include <cstddef>
#include <cuda.h>

#include "fileIO.h"
#include "filter.h"
#include "sort.h"
#include "xcorr.h"

// Provide some namespace shortcuts
using std::cout;
using std::chrono::high_resolution_clock;
using std::chrono::duration;

// ----- BEGIN MAIN PROGRAM -----
int main(int argc, char *argv[]) {
  char *fileIn = "check2.lvm"; // file name to load
  char *pushFile =
      "processed_push.csv"; // file name to which we will write the "push" data
  char *releaseFile = "processed_release.csv"; // file name to which we will
                                               // write the "release" data
  double **dataMatrix;

  // number of fields (i.e. data columns) in the .lvm file
  int numFields = 0; // (set by readLVM)

  // number of samples (i.e. data rows) in the .lvm file (set by readLVM)
  int numSamples = 0;

  int sampleFreq = 50000; // Hz

  // lower and upper cutoff for Butterworth bandpass
  double filter[2] = {150, 5000}; // [Hz]

  // start and end of template window for xcorr
  double window[2] = {0, 2}; // [ms]

  // accelerometer spacing
  double travelDist = 10; // [mm]

  int i; // index variable

  // ----- timing -----
  high_resolution_clock::time_point start;
  start = high_resolution_clock::now();
  // ----- timing -----

  // call readLVM to load data into a matrix
  dataMatrix = readLVM(fileIn, &numFields, &numSamples);

  // filter first accelerometer data
  double *filteredAcc1, *filteredAcc2;
  filteredAcc1 = filtfilt(dataMatrix[1], numSamples, sampleFreq, filter[0],
                          filter[1]); // filter acc1 data
  filteredAcc2 = filtfilt(dataMatrix[2], numSamples, sampleFreq, filter[0],
                          filter[1]); // filter acc2 data

  /// sort the push and pull tap indices
  int *pushPullIndices;
  int nLead, nTrail;
  size_t ind1, ind2;

  // returns indices of "push" (rising edges) and "release" (falling edges) in
  // array
  // values are ordered as follows: [push1, push2, ... pushN, release1,
  // release2, ... releaseN]
  pushPullIndices =
      sort(dataMatrix[0], sampleFreq, numSamples, 100, &nLead, &nTrail);

  // compute wave speed
  bool whichFirst = 0;
  double *push, *release;
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

  push = (float *)malloc(sizeof(double) * nPush);
  release = (float *)malloc(sizeof(double) * nRelease);

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
  }   // end for each push

  computeWaveSpeed(filteredAcc1, filteredAcc2, ind1, ind2, push, nPush, window,
                   sampleRate, travelDist);

  // RELEASE
  for (i = 0; i < nRelease; i++) {
    if (!whichFirst) {
      ind1 = pushPullIndices[nPush + i];
      ind2 = pushPullIndices[i + 1];
    } else {
      ind1 = pushPullIndices[nPush + i];
      ind2 = pushPullIndices[i];
    } // end conditional whichFirst
  }   // end for each release
  computeWaveSpeed(filteredAcc1, filteredAcc2, ind1, ind2, release, nRelease,
                   window, sampleRate, travelDist);

  // ----- timing -----
  high_resolution_clock::time_point end;
  duration<double, std::milli> duration_sec;
  end = high_resolution_clock::now();
  duration_sec =
      std::chrono::duration_cast<duration<double, std::milli>>(end - start);
  cout << "Execution time: " << duration_sec.count() << " ms\n";
  // ----- timing -----

  writeCSV(pushFile, push, nPush, 1);          // write out push data
  writeCSV(releaseFile, release, nRelease, 1); // write out release data

  // free all allocated memory
  free(dataMatrix);
  free(filteredAcc1);
  free(pushPullIndices);
  free(push);
  free(release);
  free(ind1);
  free(ind2);

  return 0;
} // end main