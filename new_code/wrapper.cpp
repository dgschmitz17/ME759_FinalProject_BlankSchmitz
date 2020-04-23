/*! \mainpage Authors: Blank, J. and Schmitz, D.
 * \section intro_sec Introduction
 * This is the main "wrapper" program that computes wave speed
 *
 * Function calls:  readLVM
 *                   filtfilt
 *                   sort
 *                   computeTimeDelay
 *                   writeCSV
 *
 *  Compile command (gcc):
 *  gcc wrapper.c filtfilt.c sort.c xcorr.c butterLP.c butterHP.c readLVM.c writeCSV.c -Wall -O3 -o wrapper.exe
 */

// ----- LIBRARIES ----- //
#include <cstdlib>
#include <cstdbool>
#include <iostream>
#include <chrono>
#include <ratio>
#include <cmath>
#include <cstddef>

#include "fileIO.h"
#include "filter.h"
#include "sort.h"
#include "xcorr.h"

// Provide some namespace shortcuts
using std::cout;
using std::chrono::high_resolution_clock;
using std::chrono::duration;

// ----- BEGIN MAIN PROGRAM -----
int main(int argc,char *argv[]){
    char *fileIn = "check2.lvm"; // file name to load
    char *pushFile = "processed_push.csv"; // file name to which we will write the "push" data
    char *releaseFile = "processed_release.csv"; // file name to which we will write the "release" data
    double **dataMatrix;

    int numFields = 0; // number of fields (i.e. data columns) in the .lvm file (set by readLVM)
    int numSamples = 0; // number of samples (i.e. data rows) in the .lvm file (set by readLVM)

    int sampleFreq = 50000; // Hz

    double filter[2] = {150,5000}; // lower and upper cutoff for Butterworth bandpass
    double window[2] = {0,2}; // [ms], start and end of template window for xcorr
    double travelDist = 10; // [mm], accelerometer spacing

    int i; // index variable

    // ----- timing -----
    high_resolution_clock::time_point start;
    start = high_resolution_clock::now();
    // ----- timing -----

    // call readLVM to load data into a matrix
    dataMatrix = readLVM(fileIn,&numFields,&numSamples);
   
    // filter first accelerometer data
    double *filteredAcc1, *filteredAcc2;
    filteredAcc1 = filtfilt(dataMatrix[1], numSamples, sampleFreq, filter[0], filter[1]); // filter acc1 data
    filteredAcc2 = filtfilt(dataMatrix[2], numSamples, sampleFreq, filter[0], filter[1]); // filter acc2 data

    /// sort the push and pull tap indices
    int *pushPullIndices;
    int nLead, nTrail;
    size_t ind1, ind2;

    // returns indices of "push" (rising edges) and "release" (falling edges) in array
    // values are ordered as follows: [push1, push2, ... pushN, release1, release2, ... releaseN]
    pushPullIndices = sort(dataMatrix[0], sampleFreq, numSamples, 100, &nLead, &nTrail);

    // compute wave speed
    bool whichFirst = 0;
    double *push, *release;
    int nPush, nRelease;

    // determine which comes first: push or release. Also determine how many push and release events we can use
    if(nLead > nTrail){ // push starts and ends, so throw away last push index, both push and release have length nTrail
        push = (double *) malloc(sizeof(double) * nTrail);
        nPush = nTrail;

        release = (double *) malloc(sizeof(double) * nTrail);
        nRelease = nTrail;
    }else if(nLead < nTrail){ // release starts and ends, so throw away last release index, both push and release have length nLead
        whichFirst = 1;
        push = (double *) malloc(sizeof(double) * nLead);
        nPush = nLead;

        release = (double *) malloc(sizeof(double) * nLead);
        nRelease = nLead;
    }else{ // same number of push and release, so need to determine which comes first and how many of each
        if(pushPullIndices[0] > pushPullIndices[nLead]){ // determine which is first
            whichFirst = 1;
        }

        if(!whichFirst){ // push is first, push will have one more event than release
            push = (double *) malloc(sizeof(double) * nLead);
            nPush = nLead;

            release = (double *) malloc(sizeof(double) * (nLead-1));
            nRelease = nLead - 1;
        }else{ // release is first, release will have one more event than push
            push = (double *) malloc(sizeof(double) * (nTrail-1));
            nPush = nTrail - 1;

            release = (double *) malloc(sizeof(double) * nTrail);
            nRelease = nTrail;
        }
    }// if malloc based on relative push/release length

    // comput wave speed for all push events
    for( i = 0 ; i < nPush ; i++ ){
        // determine event indices to pass to computeTimeDelay
        if(!whichFirst){
            ind1 = pushPullIndices[i];
            ind2 = pushPullIndices[nPush + i];
        }else{
            ind1 = pushPullIndices[i];
            ind2 = pushPullIndices[nPush + i + 1];
        }// end conditional whichFirst

        // wave speed is the travel distance over the delay in wave arrival times
        push[i] =  travelDist/computeTimeDelay(filteredAcc1, filteredAcc2, ind1, ind2, window, sampleFreq);
    }// end for each push

    for( i = 0 ; i < nRelease ; i++ ){
        if(!whichFirst){
            ind1 = pushPullIndices[nPush + i];
            ind2 = pushPullIndices[i + 1];
        }else{
            ind1 = pushPullIndices[nPush + i];
            ind2 = pushPullIndices[i];
        }// end conditional whichFirst

        // wave speed is the travel distance over the delay in wave arrival times
        release[i] =  travelDist/computeTimeDelay(filteredAcc1, filteredAcc2, ind1, ind2, window, sampleFreq);
    }// end for each push

    // ----- timing -----
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;
    end = high_resolution_clock::now();
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
    cout << "Execution time: ";
    cout << duration_sec.count();
    cout << "seconds\n";
    // ----- timing -----

    writeCSV(pushFile,push,nPush,1); // write out push data
    writeCSV(releaseFile,release,nRelease,1); // write out release data

    // free all allocated memory
    free(dataMatrix);
    free(filteredAcc1);
    free(pushPullIndices);
    free(push);
    free(release);

    return 0;
}// end main