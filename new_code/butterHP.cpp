/*! Authors: Blank, J. and Schmitz, D.
 * Implementation of N-th order Butterworth high-pass filter
 * Adapted from Matlab implementation by Niel Robertson: https://www.dsprelated.com/showarticle/1119.php
 * Author: Dylan Schmitz
 *
 * Inputs:  fc  -----   cut-off frequency (Hz)
 *          fs  -----   sampling frequency (Hz)
 *
 * Outputs: B   -----   numerator coefficients of the resultant transfer function
 *          A   -----   denominator coefficients of the resultant transfer function
 *
 * The function currently supports only a 2nd order filter.
 */

#include <stdlib.h>
#include <stdio.h>
#include <complex.h>
#include <math.h>
#include <iostream>
#include <cstddef>

// Provide some namespace shortcuts
using std::cout;

#define PI 3.1416
#define complex _Complex

void butterHP(int fc, int fs, double *B, double *A){
    int N = 2;
    if(fc >= fs/2){
        cout << "The high-pass cutoff frequency must be less than half the sample rate.";
        exit(100);
    }// check that cutoff is below half the sample frequency

    double Fc = fs/PI * tan(PI*fc/fs); // continuous pre-warped frequency

    double theta[2];
    double complex p_lp[2];
    double complex pa[2];
    double complex p[2];
    double b[3] = {1,-2,1};
    double K;

    int i, k;
    for( k = 0 ; k < N ; k++ ){
        // I. Find the poles of normalized analog lowpass filter
        theta[k] = (2*(k+1)-1)*PI/(2*N);

        // poles of lp with cutoff = 1 rad/s
        p_lp[k] = (-sin(theta[k])) + (cos(theta[k]))*I;

        // II. transform the poles for hpf
        pa[k] = 2*PI*Fc/p_lp[k]; // analog hp poles

        // III. Find coefs of digital filter
        p[k] = (1+pa[k]/(2*fs))/(1-pa[k]/(2*fs)); // poles by bilinear transform

    }// end for, order of filter

    // convert poles and zeros to polynomial coefficients
    double complex temp;
    A[0] = 1;
    temp =  -(p[0] + p[1]);
    A[1] = creal(temp);
    temp =  p[0] * p[1];
    A[2] = creal(temp);

    K = (A[0] - A[1] + A[2])/(b[0] - b[1] + b[2]);
    for( i = 0 ; i < 3 ; i++ ){
        B[i] = b[i]*K;
    }// end for

}// end butterLP

/*
% hp_synth.m    1/30/18 Neil Robertson
% Find the coefficients of an IIR Butterworth highpass filter using bilinear transform.
%
% N= filter order
% fc= -3 dB frequency in Hz
% fs= sample frequency in Hz
% b = numerator coefficients of digital filter
% a = denominator coefficients of digital filter
%

function [b,a]= hp_synth(N,fc,fs);

if fc>=fs/2;
   error('fc must be less than fs/2')
end


% I.  Find poles of normalized analog lowpass filter
k= 1:N;
theta= (2*k -1)*pi/(2*N);
p_lp= -sin(theta) + j*cos(theta);     % poles of lpf with cutoff = 1 rad/s


% II.  transform poles for hpf
Fc= fs/pi * tan(pi*fc/fs);             % continuous pre-warped frequency
pa= 2*pi*Fc./p_lp;                     % analog hp poles


% III.  Find coeffs of digital filter
% poles and zeros in the z plane
p= (1 + pa/(2*fs))./(1 - pa/(2*fs));   % poles by bilinear transform
q= ones(1,N);                          % zeros at z = 1 (f= 0)


% convert poles and zeros to polynomial coeffs
a= poly(p);                   % convert poles to polynomial coeffs a
a= real(a);
b= poly(q);                   % convert zeros to polynomial coeffs b


% amplitude scale factor for gain = 1 at f = fs/2  (z = -1)
m= 0:N;
K= sum((-1).^m .*a)/sum((-1).^m .*b);        % amplitude scale factor
b= K*b;
*/