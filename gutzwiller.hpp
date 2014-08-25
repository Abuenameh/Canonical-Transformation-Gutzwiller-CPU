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

typedef complex<double> doublecomplex;

#define L 50
#define nmax 5
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

inline int mod(int i) {
	return (i + L) % L;
}

inline double g(int n, int m) {
	return sqrt(1.0*(n + 1) * m);
}

extern vector<double> nu;

inline double eps(vector<double> U, int i, int j, int n, int m) {
	return n * U[i] - (m - 1) * U[j] + nu[i] - nu[j];
}

double Encfunc(unsigned ndim, const double *x, double *grad, void *data);
double Encnnfunc(unsigned ndim, const double *x, double *grad, void *data);
double Encfunc2(unsigned ndim, const double *x, double *grad, void *data);
double Ecfunc(unsigned ndim, const double *x, double *grad, void *data);




#endif	/* GUTZWILLER_HPP */

