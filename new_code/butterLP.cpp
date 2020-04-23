/*! Authors: Blank, J. and Schmitz, D.
 * Implementation of N-th order Butterworth low-pass filter
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

#include <cstdlib>
#include <complex>
#include <cmath>
#include <iostream>
#include <cstddef>

// Provide some namespace shortcuts
using std::cout;

#define PI 3.1416
#define complex _Complex

void butterLP(int fc, int fs, double *B, double *A){
    int N = 2;
    if(fc >= fs/2){
        cout << "The low-pass cutoff frequency must be less than half the sample rate.";
        exit(100);
    }// check that cutoff is below half the sample frequency

    double Fc = fs/PI * tan(PI*fc/fs); // continuous pre-warped frequency
    double complex num;
    double complex den;

    // I. Find the poles of the analog filter
    double theta[2];
    double pa_Re[2];
    double pa_Im[2];
    double complex p[2];
    double b[3] = {1,2,1};
    double K;

    int i, k;
    for( k = 0 ; k < N ; k++ ){
        theta[k] = (2*(k+1)-1)*PI/(2*N);

        // poles of filter with cutoff = 1 rad/s
        // II. Scale poles in frequency by 2*pi*fc
        pa_Re[k] = 2*PI*Fc * (-sin(theta[k]));
        pa_Im[k] = 2*PI*Fc * (cos(theta[k]));

        // III. Find coefs of digital filter
        num = (1 + pa_Re[k]/2/fs) + (pa_Im[k]/2/fs)*I;
        den = (1 - pa_Re[k]/2/fs) - (pa_Im[k]/2/fs)*I;

        p[k] = num/den; // poles by bilinear transform

    }// end for, order of filter

    // convert poles and zeros to polynomial coefficients
    double complex temp;
    A[0] = 1;
    temp =  -(p[0] + p[1]);
    A[1] = creal(temp);
    temp =  p[0] * p[1];
    A[2] = creal(temp);

    K = (A[0] + A[1] + A[2])/(b[0] + b[1] + b[2]);
    for( i = 0 ; i < 3 ; i++ ){
        B[i] = b[i]*K;
    }// end for

}// end butterLP

/*
function [b,a]= butter_synth(N,fc,fs)

if fc>=fs/2
   error('fc must be less than fs/2')
end
// I.  Find poles of analog filter
k= 1:N;
theta= (2*k -1)*pi/(2*N);
pa= -sin(theta) + 1j*cos(theta);     // poles of filter with cutoff = 1 rad/s

// II.  scale poles in frequency
Fc= fs/pi * tan(pi*fc/fs);          // continuous pre-warped frequency
pa= pa*2*pi*Fc;                     // scale poles by 2*pi*Fc

// III.  Find coeffs of digital filter
// poles and zeros in the z plane
p= (1 + pa/(2*fs))./(1 - pa/(2*fs));      // poles by bilinear transform
q= -ones(1,N);                   // zeros

// convert poles and zeros to polynomial coeffs
a= poly(p);                   // convert poles to polynomial coeffs a
a= real(a);
b= poly(q);                   // convert zeros to polynomial coeffs b
K= sum(a)/sum(b);             // amplitude scale factor
b= K*b;
*/