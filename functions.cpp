#include "gutzwiller.hpp"

double Efunc(unsigned ndim, const double *x, double *grad, void *data) {
    parameters* parms = static_cast<parameters*> (data);
    vector<double> U = parms->U;
    double mu = parms->mu;
    vector<double> J = parms->J;
    double theta = parms->theta;

    double costh = cos(theta);

    doublecomplex Ec = 0;

    const doublecomplex * f[L];
    vector<double> norm2(L, 0);
    for (int i = 0; i < L; i++) {
        f[i] = reinterpret_cast<const doublecomplex*> (&x[2 * i * dim]);
        for (int n = 0; n <= nmax; n++) {
            norm2[i] += norm(f[i][n]);
        }
    }

    vector<doublecomplex> E0s(L, 0), E1j1s(L, 0), E1j2s(L, 0);


    for (int i = 0; i < L; i++) {

        int j1 = mod(i - 1);
        int j2 = mod(i + 1);

        doublecomplex E0 = 0;
        doublecomplex E1j1 = 0;
        doublecomplex E1j2 = 0;

        for (int n = 0; n <= nmax; n++) {

            E0 += (0.5 * U[i] * n * (n - 1) - mu * n) * ~f[i][n] * f[i][n];

            if (n < nmax) {
                for (int m = 1; m <= nmax; m++) {
                    E1j1 += -J[j1] * costh * g(n, m) * ~f[i][n + 1] * ~f[j1][m - 1]
                            * f[i][n] * f[j1][m];
                    E1j2 += -J[i] * costh * g(n, m) * ~f[i][n + 1] * ~f[j2][m - 1]
                            * f[i][n] * f[j2][m];
                }
            }

        }

        Ec += E0 / norm2[i];
        Ec += E1j1 / (norm2[i] * norm2[j1]);
        Ec += E1j2 / (norm2[i] * norm2[j2]);

        E0s[i] += E0;
        E1j1s[i] += E1j1;
        E1j2s[i] += E1j2;

    }

    if (grad) {
        for (int i = 0; i < L; i++) {
            int j1 = mod(i - 1);
            int j2 = mod(i + 1);

            for (int n = 0; n <= nmax; n++) {
        doublecomplex E0df = 0;
        doublecomplex E1j1df = 0;
        doublecomplex E1j2df = 0;

                E0df += (0.5 * U[i] * n * (n - 1) - mu * n) * f[i][n];

                if (n < nmax) {
                    for (int m = 0; m < nmax; m++) {
                        E1j1df += -J[j1] * costh * g(n, m + 1) * ~f[j1][m + 1] * f[j1][m]
                                * f[i][n + 1];
                        E1j2df += -J[i] * costh * g(n, m + 1) * ~f[j2][m + 1] * f[j2][m]
                                * f[i][n + 1];
                    }
                }
                if (n > 0) {
                    for (int m = 1; m <= nmax; m++) {
                        E1j1df += -J[j1] * costh * g(n - 1, m) * ~f[j1][m - 1] * f[j1][m]
                                * f[i][n - 1];
                        E1j2df += -J[i] * costh * g(n - 1, m) * ~f[j2][m - 1] * f[j2][m]
                                * f[i][n - 1];
                    }
                }
                
                doublecomplex Edf = 0;

                Edf += (E0df * norm2[i] - E0s[i] * f[i][n]) / (norm2[i] * norm2[i]);

                Edf += (E1j1df * norm2[i] * norm2[j1]
                        - (E1j1s[i] + E1j2s[j1]) * f[i][n] * norm2[j1])
                        / (norm2[i] * norm2[i] * norm2[j1] * norm2[j1]);
                Edf += (E1j2df * norm2[i] * norm2[j2]
                        - (E1j2s[i] + E1j1s[j2]) * f[i][n] * norm2[j2])
                        / (norm2[i] * norm2[i] * norm2[j2] * norm2[j2]);

                int k = i * dim + n;
                grad[2 * k] = 2 * Edf.real();
                grad[2 * k + 1] = 2 * Edf.imag();

            }
        }
    }
    
    return Ec.real();
}

void norm2s(unsigned m, double *result, unsigned ndim, const double* x,
	double* grad, void* data) {

	const doublecomplex * f[L];
	for (int i = 0; i < L; i++) {
		f[i] = reinterpret_cast<const doublecomplex*>(&x[2 * i * dim]);
	}

	for (int i = 0; i < L; i++) {
		result[i] = 0;
		for (int n = 0; n <= nmax; n++) {
			result[i] += norm(f[i][n]);
		}
		result[i] -= 1;
	}

	if (grad) {
		for (int i = 0; i < L; i++) {
			for (int j = 0; j < ndim; j++) {
				grad[i * ndim + j] = 2 * x[i * ndim + j];
			}
		}
	}
}

