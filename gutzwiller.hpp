/* 
 * File:   gutzwiller.hpp
 * Author: Abuenameh
 *
 * Created on August 10, 2014, 10:45 PM
 */

#ifndef GUTZWILLER_HPP
#define	GUTZWILLER_HPP

#include <complex>
#include <vector>
#include <iostream>

using namespace std;

#ifdef __NVCC__
#include "cudacomplex.hpp"
#define G_HOST __host__
#define G_HOSTDEVICE __host__ __device__
#else
typedef complex<double> doublecomplex;
#define G_HOST
#define G_HOSTDEVICE
#endif

#define L 50
#define nmax 7
#define dim (nmax+1)

template<class T>
complex<T> operator~(const complex<T> a) {
	return conj(a);
}

struct parameters {
    bool canonical;
	vector<double> U;
	double mu;
	vector<double> J;
    double theta;
//	double* U;
//	double mu;
//	double* J;
};

struct phase_parameters {
    double theta;
    bool canonical;
};

struct device_parameters {
	double* U;
	double mu;
	double* J;
    double theta;
};

G_HOSTDEVICE inline int mod(int i) {
	return (i + L) % L;
}

G_HOSTDEVICE inline double g(int n, int m) {
	return sqrt(1.0*(n + 1) * m);
}

G_HOST inline double eps(vector<double> U, int i, int j, int n, int m) {
	return n * U[i] - (m - 1) * U[j];
}

double Encfunc(unsigned ndim, const double *x, double *grad, void *data);
double Ecfunc(unsigned ndim, const double *x, double *grad, void *data);

#ifdef __NVCC__
__device__ inline double eps(double* U, int i, int j, int n, int m) {
	return n * U[i] - (m - 1) * U[j];
}
#endif



#endif	/* GUTZWILLER_HPP */

