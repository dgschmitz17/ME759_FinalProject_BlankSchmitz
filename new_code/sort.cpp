/*! Authors: Blank, J. and Schmitz, D.
 * This function sorts the tap signal into separate push and pull events that will be used in the xcorr function
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <iostream>
#include <cstddef>
#include "sort.h"

// finds the max of a given array
int max_jon(const double* arr, size_t size){
    // initialize the max
    int max = 0; 
    // find the maximum value in the array
    for(int i=0; i<size; i++){
        if (arr[i]>arr[max]){
            max = i;
        }
    }
    return max;
}

// computed the cross-correlation given two signals, a window, the sampling time and returns
// the correlation coefficient and time delay
int *sort(const double* signalRef, size_t ts, size_t n, size_t tapRate, int* lead, int* trail){

    // initialize the new tap signal
    double* signal = (double*)malloc(n*sizeof(double));

    // get rid of aberrant negative values
    for (int i=1; i<n-1; i++){
        if (signalRef[i]<0){
            signal[i]=(signalRef[i-1]+signalRef[i+1])/2;
        }
    }

    // calculate the data points per tap
    double m = ts/tapRate;
    // calculate the total number of push and pull taps
    double nTaps = n*(1/m);
    // create an array of booleans the size of signalRef
    int* extended = (int*)malloc(n*sizeof(int));
    int* retracted = (int*)malloc(n*sizeof(int));
    int* leading = (int*)malloc((int)n*sizeof(int));
    int* trailing = (int*)malloc((int)n*sizeof(int));
    // find pulse edges of the tap signal
    int j = 0; int k = 0; int flag = 0;
    double thresh = signalRef[max_jon(signalRef,n)]/2;
    for (int i=0; i<n-1; i++){
        // store the extended and retracted tapper data
        if (signalRef[i]>thresh){extended[i]=1; retracted[i]=0;}
        if (signalRef[i]<thresh){extended[i]=0; retracted[i]=1;}
    }
    for (int i=0; i<n-1; i++){
        if (i<n-1){
            // store the leading and trailing indices
            if ((extended[i+1]-extended[i])>0){flag=1;}
            if (flag==1){leading[j]=i; j++; flag=0;}    
            if ((retracted[i+1]-retracted[i])>0){flag=1;}
            if (flag==1){trailing[k]=i; k++; flag=0;}
        }
    }
    free(extended);
    free(retracted);

    // dynamically allocate memory to store leading and trailing indices
    int nLeading = 0; int nTrailing = 0;
    for (int i=0; i<nTaps/2; i++){
        if (&leading[i] != NULL){nLeading++;}
        if (&trailing[i] != NULL){nTrailing++;}
    }
    int* pushPullIndices = (int*)malloc((nTrailing+nLeading)*sizeof(int));
    if (nLeading>nTrailing){
        for (int i=0; i<nLeading; i++){
            pushPullIndices[i] = leading[i];
            if (i==nLeading-1){
                // skip this indices
            } else {
                pushPullIndices[nLeading+i] = trailing[i];
            }
        }
    } else if (nLeading<nTrailing) {
        for (int i=0; i<nTrailing; i++){
            pushPullIndices[nLeading+i] = trailing[i];
            if (i==nLeading + 1){
                // skip this indices
            } else {
                pushPullIndices[i] = leading[i];
            }
        }
    } else {
        for (int i=0; i<nLeading; i++){
            pushPullIndices[i] = leading[i];
            pushPullIndices[nLeading+i] = trailing[i];
        }
    }

    free(leading);
    free(trailing);

    *lead = nLeading;
    *trail = nTrailing;
    return pushPullIndices;
}