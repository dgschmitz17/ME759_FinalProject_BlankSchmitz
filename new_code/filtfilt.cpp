/*! Authors: Blank, J. and Schmitz, D.
 * This is the zero-phase digital filtering function (inspired by Matlab's filtfilt)
 */

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <cstddef>
#include "filter.h"

double *filtfilt(double *unfiltered, int N, int fs, double low, double high){
    double *filtered = (double *) malloc(sizeof(double) * (N));
    double *temp1 = (double *) malloc(sizeof(double) * (N));
    double *temp2 = (double *) malloc(sizeof(double) * (N));
    // Get the filter coefficients
    double B[3];
    double A[3];
    double D[3];
    double C[3];
    butterLP(high,fs,B,A); // the low-pass filter gets the "high" cutoff
    butterHP(low,fs,D,C); // the high-pass filter gets the "low" cutoff

    // pass data through low-pass filter forwards
    int i, j;
    for( i = 0 ; i < N ; i++ ){
        temp1[i] = B[0] * unfiltered[i];
        for( j = 1 ; j < 3 ; j++ ){
            if(i >= j){
                temp1[i] += B[j]*unfiltered[i-j] - A[j]*temp1[i-j];
            }//end if
        }// end for j
    }// end for i

    // pass data through low-pass filter backwards
    for( i = 0 ; i < N ; i++ ){
        temp2[(N-1)-i] = B[0] * temp1[N-1-i];
        for( j = 1 ; j < 3 ; j++ ){
            if(i >= j){
                temp2[(N-1)-i] += B[j]*temp1[(N-1)-i+j] - A[j]*temp2[(N-1)-i+j];
            }//end if
        }// end for j
    }// end for i

    // pass data through high-pass filter forwards
    for( i = 0 ; i < N ; i++ ){
        temp1[i] = D[0] * temp2[i];
        for( j = 1 ; j < 3 ; j++ ){
            if(i >= j){
                temp1[i] += D[j]*temp2[i-j] - C[j]*temp1[i-j];
            }//end if
        }// end for j
    }// end for i

    // pass data through high-pass filter backwards
    for( i = 0 ; i < N ; i++ ){
        filtered[(N-1)-i] = D[0] * temp1[N-1-i];
        for( j = 1 ; j < 3 ; j++ ){
            if(i >= j){
                filtered[(N-1)-i] += D[j]*temp1[(N-1)-i+j] - C[j]*filtered[(N-1)-i+j];
            }//end if
        }// end for j
    }// end for i

   free(temp1);
   free(temp2);

    return filtered;

}// end filtfilt