/*! Authors: Blank, J. and Schmitz, D.
 * This function sorts the tap signal into separate push and pull events that
 * will be used in the xcorr function
 */

#include <cstdlib>
#include <cstdbool>
#include <iostream>
#include <cstddef>
#include "sort.h"

// finds the max of a given array
int max_jon(const float *arr, size_t size) {
  // initialize the max
  int max = 0;
  // find the maximum value in the array
  for (size_t i = 0; i < size; i++) {
    if (arr[i] > arr[max]) {
      max = i;
    }
  }
  return max;
}

// computed the cross-correlation given two signals, a window, the sampling time
// and returns
// the correlation coefficient and time delay
size_t *sort(const float *signalRef, size_t ts, size_t n, size_t tapRate,
             size_t *lead, size_t *trail) {

  // initialize the new tap signal
  float *signal = new float[n];

  // get rid of aberrant negative values
  for (size_t i = 1; i < n - 1; i++) {
    if (signalRef[i] < 0) {
      signal[i] = (signalRef[i - 1] + signalRef[i + 1]) / 2;
    } else {
      signal[i] = signalRef[i];
    }
  }

  // calculate the data points per tap
  float m = ts / tapRate;
  // calculate the total number of push and pull taps
  float nTaps = n * (1 / m); // his value is 6,000 for check2.lvm
  // std::cout << nTaps / 2 << "\n";
  std::cout << nTaps;
  
  // create an array of booleans the size of signalRef
  int *extended = new int[n];
  int *retracted = new int[n];
  int *leading = new int[n];
  int *trailing = new int[n];
  // find pulse edges of the tap signal
  int j = 0;
  int k = 0;
  int flag = 0;
  float thresh = signal[max_jon(signal, n)] / 2;
  for (size_t i = 0; i < n; i++) {
    // store the extended and retracted tapper data
    if (signal[i] > thresh) {
      extended[i] = 1;
      retracted[i] = 0;
    }
    if (signal[i] < thresh) {
      extended[i] = 0;
      retracted[i] = 1;
    }
  }
  for (size_t i = 0; i < n - 1; i++) {
    if (i < n - 1) {
      // store the leading and trailing indices
      if ((extended[i + 1] - extended[i]) > 0) {
        flag = 1;
      }
      if (flag == 1) {
        leading[j] = i;
        j++;
        flag = 0;
      }
      if ((retracted[i + 1] - retracted[i]) > 0) {
        flag = 1;
      }
      if (flag == 1) {
        trailing[k] = i;
        k++;
        flag = 0;
      }
    }
  }
  //std::cout << k << "\n";
  //std::cout << j << "\n";
  delete[] extended;
  delete[] retracted;

  // dynamically allocate memory to store leading and trailing indices
  int nLeading = 0;
  int nTrailing = 0;
  
  // this is where the problem is
  for (size_t i = 0; i < nTaps / 2; i++) {
    if (&leading[i] != NULL) {
      nLeading++;
    }
    //std::cout << nLeading << "\n";
    if (&trailing[i] != NULL) {
      nTrailing++;
    }
  }
  // std::cout << nLeading << "\n";
  // std::cout << nTrailing << "\n";
  size_t *pushPullIndices = new size_t[nTrailing + nLeading];
  if (nLeading > nTrailing) {
    for (int i = 0; i < nLeading; i++) {
      pushPullIndices[i] = leading[i];
      if (i == nLeading - 1) {
        // skip this indices
      } else {
        pushPullIndices[nLeading + i] = trailing[i];
      }
    }
  } else if (nLeading < nTrailing) {
    for (int i = 0; i < nTrailing; i++) {
      pushPullIndices[nLeading + i] = trailing[i];
      if (i == nLeading + 1) {
      // if (i == nLeading) {
        // skip this indices
      } else {
        pushPullIndices[i] = leading[i];
      }
    }
  } else {
    for (int i = 0; i < nLeading; i++) {
      pushPullIndices[i] = leading[i];
      pushPullIndices[nLeading + i] = trailing[i];
    }
  }

  delete[] leading;
  delete[] trailing;

  *lead = nLeading;
  *trail = nTrailing;
  return pushPullIndices;
}