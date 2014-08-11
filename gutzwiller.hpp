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

#define L 3
#define nmax 5
#define dim (nmax+1)

template<class T>
complex<T> operator~(const complex<T> a) {
	return conj(a);
}

struct parameters {
	vector<double> U;
	double mu;
	vector<double> J;
    double theta;
//	double* U;
//	double mu;
//	double* J;
};

inline int mod(int i) {
	return (i + L) % L;
}

inline double g(int n, int m) {
	return sqrt((n + 1) * m);
}

inline double eps(vector<double> U, int i, int j, int n, int m) {
	return n * U[i] - (m - 1) * U[j];
}



#endif	/* GUTZWILLER_HPP */

