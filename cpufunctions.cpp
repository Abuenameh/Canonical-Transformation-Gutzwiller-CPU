#ifndef __NVCC__

#include "gutzwiller.hpp"

double Efunc(unsigned ndim, const double *x, double *grad, void *data) {
	parameters* parms = static_cast<parameters*>(data);
	vector<double> U = parms->U;
	double mu = parms->mu;
	vector<double> J = parms->J;

	doublecomplex Ec = 0;

	const doublecomplex * f[L];
	for (int i = 0; i < L; i++) {
		f[i] = reinterpret_cast<const doublecomplex*>(&x[2 * i * dim]);
	}

	for (int i = 0; i < L; i++) {
		int j1 = mod(i - 1);
		int j2 = mod(i + 1);
		int k1 = mod(i - 2);
		int k2 = mod(i + 2);
		for (int n = 0; n <= nmax; n++) {
			int k = i * dim + n;
			int l1 = j1 * dim + n;
			int l2 = j2 * dim + n;

			Ec += (0.5 * U[i] * n * (n - 1) - mu * n) * ~f[i][n] * f[i][n];

			if (n < nmax) {
				Ec += -J[j1] * g(n, n + 1) * ~f[i][n + 1] * ~f[j1][n] * f[i][n]
					* f[j1][n + 1];
				Ec += -J[i] * g(n, n + 1) * ~f[i][n + 1] * ~f[j2][n] * f[i][n]
					* f[j2][n + 1];

                if (n > 0) {
				Ec +=
					0.5 * J[j1] * J[j1] * g(n, n) * g(n - 1, n + 1)
						* ~f[i][n + 1] * ~f[j1][n - 1] * f[i][n - 1]
						* f[j1][n + 1]
						* (1 / eps(U, i, j1, n, n)
							- 1 / eps(U, i, j1, n - 1, n + 1));
				Ec +=
					0.5 * J[i] * J[i] * g(n, n) * g(n - 1, n + 1) * ~f[i][n + 1]
						* ~f[j2][n - 1] * f[i][n - 1] * f[j2][n + 1]
						* (1 / eps(U, i, j2, n, n)
							- 1 / eps(U, i, j2, n - 1, n + 1));
                }

				for (int m = 1; m <= nmax; m++) {
					if (n != m - 1) {
						Ec += 0.5 * (J[j1] * J[j1] / eps(U, i, j1, n, m))
							* g(n, m) * g(m - 1, n + 1)
							* (~f[i][n + 1] * ~f[j1][m - 1] * f[i][n + 1]
								* f[j1][m - 1]
								- ~f[i][n] * ~f[j1][m] * f[i][n] * f[j1][m]);
						Ec += 0.5 * (J[i] * J[i] / eps(U, i, j2, n, m))
							* g(n, m) * g(m - 1, n + 1)
							* (~f[i][n + 1] * ~f[j2][m - 1] * f[i][n + 1]
								* f[j2][m - 1]
								- ~f[i][n] * ~f[j2][m] * f[i][n] * f[j2][m]);
					}
				}

                if (n > 0) {
				Ec += 0.5 * (J[j1] * J[i] / eps(U, i, j1, n, n)) * g(n, n)
					* g(n - 1, n + 1) * ~f[i][n + 1] * ~f[j1][n - 1] * ~f[j2][n]
					* f[i][n - 1] * f[j1][n] * f[j2][n + 1];
				Ec += 0.5 * (J[i] * J[j1] / eps(U, i, j2, n, n)) * g(n, n)
					* g(n - 1, n + 1) * ~f[i][n + 1] * ~f[j2][n - 1] * ~f[j1][n]
					* f[i][n - 1] * f[j2][n] * f[j1][n + 1];
				Ec += 0.5 * (J[j1] * J[k1] / eps(U, i, j1, n, n)) * g(n, n)
					* g(n - 1, n + 1) * ~f[i][n + 1] * ~f[j1][n - 1] * ~f[k1][n]
					* f[i][n] * f[j1][n + 1] * f[k1][n - 1];
				Ec += 0.5 * (J[i] * J[j2] / eps(U, i, j2, n, n)) * g(n, n)
					* g(n - 1, n + 1) * ~f[i][n + 1] * ~f[j2][n - 1] * ~f[k2][n]
					* f[i][n] * f[j2][n + 1] * f[k2][n - 1];
				Ec -= 0.5 * (J[j1] * J[i] / eps(U, i, j1, n - 1, n + 1))
					* g(n, n) * g(n - 1, n + 1) * ~f[i][n + 1] * ~f[j1][n]
					* ~f[j2][n - 1] * f[i][n - 1] * f[j1][n + 1] * f[j2][n];
				Ec -= 0.5 * (J[i] * J[j1] / eps(U, i, j2, n - 1, n + 1))
					* g(n, n) * g(n - 1, n + 1) * ~f[i][n + 1] * ~f[j2][n]
					* ~f[j1][n - 1] * f[i][n - 1] * f[j2][n + 1] * f[j1][n];
				Ec -= 0.5 * (J[j1] * J[k1] / eps(U, i, j1, n - 1, n + 1))
					* g(n, n) * g(n - 1, n + 1) * ~f[i][n] * ~f[j1][n - 1]
					* ~f[k1][n + 1] * f[i][n - 1] * f[j1][n + 1] * f[k1][n];
				Ec -= 0.5 * (J[i] * J[j2] / eps(U, i, j2, n - 1, n + 1))
					* g(n, n) * g(n - 1, n + 1) * ~f[i][n] * ~f[j2][n - 1]
					* ~f[k2][n + 1] * f[i][n - 1] * f[j2][n + 1] * f[k2][n];
                }
			}

			for (int m = 1; m <= nmax; m++) {
				if (n != m - 1 && n < nmax) {
					Ec += 0.5 * (J[j1] * J[i] / eps(U, i, j1, n, m)) * g(n, m)
						* g(m - 1, n + 1) * ~f[i][n + 1] * ~f[j1][m - 1]
						* ~f[j2][m] * f[i][n + 1] * f[j1][m] * f[j2][m - 1];
					Ec += 0.5 * (J[i] * J[j1] / eps(U, i, j2, n, m)) * g(n, m)
						* g(m - 1, n + 1) * ~f[i][n + 1] * ~f[j2][m - 1]
						* ~f[j1][m] * f[i][n + 1] * f[j2][m] * f[j1][m - 1];
					Ec += 0.5 * (J[j1] * J[k1] / eps(U, i, j1, n, m)) * g(n, m)
						* g(m - 1, n + 1) * ~f[i][n + 1] * ~f[j1][m - 1]
						* ~f[k1][n] * f[i][n] * f[j1][m - 1] * f[k1][n + 1];
					Ec += 0.5 * (J[i] * J[j2] / eps(U, i, j2, n, m)) * g(n, m)
						* g(m - 1, n + 1) * ~f[i][n + 1] * ~f[j2][m - 1]
						* ~f[k2][n] * f[i][n] * f[j2][m - 1] * f[k2][n + 1];
					Ec -= 0.5 * (J[j1] * J[i] / eps(U, i, j1, n, m)) * g(n, m)
						* g(m - 1, n + 1) * ~f[i][n] * ~f[j1][m - 1] * ~f[j2][m]
						* f[i][n] * f[j1][m] * f[j2][m - 1];
					Ec -= 0.5 * (J[i] * J[j1] / eps(U, i, j2, n, m)) * g(n, m)
						* g(m - 1, n + 1) * ~f[i][n] * ~f[j2][m - 1] * ~f[j1][m]
						* f[i][n] * f[j2][m] * f[j1][m - 1];
					Ec -= 0.5 * (J[j1] * J[k1] / eps(U, i, j1, n, m)) * g(n, m)
						* g(m - 1, n + 1) * ~f[i][n + 1] * ~f[j1][m] * ~f[k1][n]
						* f[i][n] * f[j1][m] * f[k1][n + 1];
					Ec -= 0.5 * (J[i] * J[j2] / eps(U, i, j2, n, m)) * g(n, m)
						* g(m - 1, n + 1) * ~f[i][n + 1] * ~f[j2][m] * ~f[k2][n]
						* f[i][n] * f[j2][m] * f[k2][n + 1];
				}
			}
		}
	}

//    cout << Ec.real() << endl;
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

#endif
