#include "gutzwiller.hpp"
#include "cudautils.hpp"

#ifdef __NVCC__

__device__ double atomicAdd(double* address, double val) {
	unsigned long long int* address_as_ull = (unsigned long long int*) address;
	unsigned long long int old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,
			__double_as_longlong(val + __longlong_as_double(assumed)));
	} while (assumed != old);
	return __longlong_as_double(old);
}

__global__ void Efuncker(unsigned ndim, const double *x, double* fval,
	double *grad, void *data) {

	const int i = threadIdx.x;
	if (i >= L) {
		return;
	}

	device_parameters* parms = static_cast<device_parameters*>(data);
	double* U = parms->U;
	double mu = parms->mu;
	double* J = parms->J;

	doublecomplex Ec = doublecomplex::zero();

	const doublecomplex * f[L];
	for (int i = 0; i < L; i++) {
		f[i] = reinterpret_cast<const doublecomplex*>(&x[2 * i * dim]);
	}

//	for (int i = 0; i < L; i++) {
	int j1 = mod(i - 1);
	int j2 = mod(i + 1);
	int k1 = mod(i - 2);
	int k2 = mod(i + 2);
		for (int n = 0; n <= nmax; n++) {
	int k = i * dim + n;
	int l1 = j1 * dim + n;
	int l2 = j2 * dim + n;

	Ec = Ec + (0.5 * U[i] * n * (n - 1) - mu * n) * ~f[i][n] * f[i][n];

	if (n < nmax) {
		Ec = Ec
			+ -J[j1] * g(n, n + 1) * ~f[i][n + 1] * ~f[j1][n] * f[i][n]
				* f[j1][n + 1];
		Ec = Ec
			+ -J[i] * g(n, n + 1) * ~f[i][n + 1] * ~f[j2][n] * f[i][n]
				* f[j2][n + 1];

		if (n > 0) {
			Ec =
				Ec
					+ 0.5 * J[j1] * J[j1] * g(n, n) * g(n - 1, n + 1)
						* ~f[i][n + 1] * ~f[j1][n - 1] * f[i][n - 1]
						* f[j1][n + 1]
						* (1 / eps(U, i, j1, n, n)
							- 1 / eps(U, i, j1, n - 1, n + 1));
			Ec =
				Ec
					+ 0.5 * J[i] * J[i] * g(n, n) * g(n - 1, n + 1)
						* ~f[i][n + 1] * ~f[j2][n - 1] * f[i][n - 1]
						* f[j2][n + 1]
						* (1 / eps(U, i, j2, n, n)
							- 1 / eps(U, i, j2, n - 1, n + 1));
		}

		for (int m = 1; m <= nmax; m++) {
			if (n != m - 1) {
				Ec = Ec
					+ 0.5 * (J[j1] * J[j1] / eps(U, i, j1, n, m)) * g(n, m)
						* g(m - 1, n + 1)
						* (~f[i][n + 1] * ~f[j1][m - 1] * f[i][n + 1]
							* f[j1][m - 1]
							- ~f[i][n] * ~f[j1][m] * f[i][n] * f[j1][m]);
				Ec = Ec
					+ 0.5 * (J[i] * J[i] / eps(U, i, j2, n, m)) * g(n, m)
						* g(m - 1, n + 1)
						* (~f[i][n + 1] * ~f[j2][m - 1] * f[i][n + 1]
							* f[j2][m - 1]
							- ~f[i][n] * ~f[j2][m] * f[i][n] * f[j2][m]);
			}
		}

		if (n > 0) {
			Ec = Ec
				+ 0.5 * (J[j1] * J[i] / eps(U, i, j1, n, n)) * g(n, n)
					* g(n - 1, n + 1) * ~f[i][n + 1] * ~f[j1][n - 1] * ~f[j2][n]
					* f[i][n - 1] * f[j1][n] * f[j2][n + 1];
			Ec = Ec
				+ 0.5 * (J[i] * J[j1] / eps(U, i, j2, n, n)) * g(n, n)
					* g(n - 1, n + 1) * ~f[i][n + 1] * ~f[j2][n - 1] * ~f[j1][n]
					* f[i][n - 1] * f[j2][n] * f[j1][n + 1];
			Ec = Ec
				+ 0.5 * (J[j1] * J[k1] / eps(U, i, j1, n, n)) * g(n, n)
					* g(n - 1, n + 1) * ~f[i][n + 1] * ~f[j1][n - 1] * ~f[k1][n]
					* f[i][n] * f[j1][n + 1] * f[k1][n - 1];
			Ec = Ec
				+ 0.5 * (J[i] * J[j2] / eps(U, i, j2, n, n)) * g(n, n)
					* g(n - 1, n + 1) * ~f[i][n + 1] * ~f[j2][n - 1] * ~f[k2][n]
					* f[i][n] * f[j2][n + 1] * f[k2][n - 1];
			Ec = Ec
				- 0.5 * (J[j1] * J[i] / eps(U, i, j1, n - 1, n + 1)) * g(n, n)
					* g(n - 1, n + 1) * ~f[i][n + 1] * ~f[j1][n] * ~f[j2][n - 1]
					* f[i][n - 1] * f[j1][n + 1] * f[j2][n];
			Ec = Ec
				- 0.5 * (J[i] * J[j1] / eps(U, i, j2, n - 1, n + 1)) * g(n, n)
					* g(n - 1, n + 1) * ~f[i][n + 1] * ~f[j2][n] * ~f[j1][n - 1]
					* f[i][n - 1] * f[j2][n + 1] * f[j1][n];
			Ec = Ec
				- 0.5 * (J[j1] * J[k1] / eps(U, i, j1, n - 1, n + 1)) * g(n, n)
					* g(n - 1, n + 1) * ~f[i][n] * ~f[j1][n - 1] * ~f[k1][n + 1]
					* f[i][n - 1] * f[j1][n + 1] * f[k1][n];
			Ec = Ec
				- 0.5 * (J[i] * J[j2] / eps(U, i, j2, n - 1, n + 1)) * g(n, n)
					* g(n - 1, n + 1) * ~f[i][n] * ~f[j2][n - 1] * ~f[k2][n + 1]
					* f[i][n - 1] * f[j2][n + 1] * f[k2][n];
		}

		for (int m = 1; m <= nmax; m++) {
			if (n != m - 1 && n < nmax) {
				Ec = Ec
					+ 0.5 * (J[j1] * J[i] / eps(U, i, j1, n, m)) * g(n, m)
						* g(m - 1, n + 1) * ~f[i][n + 1] * ~f[j1][m - 1]
						* ~f[j2][m] * f[i][n + 1] * f[j1][m] * f[j2][m - 1];
				Ec = Ec
					+ 0.5 * (J[i] * J[j1] / eps(U, i, j2, n, m)) * g(n, m)
						* g(m - 1, n + 1) * ~f[i][n + 1] * ~f[j2][m - 1]
						* ~f[j1][m] * f[i][n + 1] * f[j2][m] * f[j1][m - 1];
				Ec = Ec
					+ 0.5 * (J[j1] * J[k1] / eps(U, i, j1, n, m)) * g(n, m)
						* g(m - 1, n + 1) * ~f[i][n + 1] * ~f[j1][m - 1]
						* ~f[k1][n] * f[i][n] * f[j1][m - 1] * f[k1][n + 1];
				Ec = Ec
					+ 0.5 * (J[i] * J[j2] / eps(U, i, j2, n, m)) * g(n, m)
						* g(m - 1, n + 1) * ~f[i][n + 1] * ~f[j2][m - 1]
						* ~f[k2][n] * f[i][n] * f[j2][m - 1] * f[k2][n + 1];
				Ec = Ec
					- 0.5 * (J[j1] * J[i] / eps(U, i, j1, n, m)) * g(n, m)
						* g(m - 1, n + 1) * ~f[i][n] * ~f[j1][m - 1] * ~f[j2][m]
						* f[i][n] * f[j1][m] * f[j2][m - 1];
				Ec = Ec
					- 0.5 * (J[i] * J[j1] / eps(U, i, j2, n, m)) * g(n, m)
						* g(m - 1, n + 1) * ~f[i][n] * ~f[j2][m - 1] * ~f[j1][m]
						* f[i][n] * f[j2][m] * f[j1][m - 1];
				Ec = Ec
					- 0.5 * (J[j1] * J[k1] / eps(U, i, j1, n, m)) * g(n, m)
						* g(m - 1, n + 1) * ~f[i][n + 1] * ~f[j1][m] * ~f[k1][n]
						* f[i][n] * f[j1][m] * f[k1][n + 1];
				Ec = Ec
					- 0.5 * (J[i] * J[j2] / eps(U, i, j2, n, m)) * g(n, m)
						* g(m - 1, n + 1) * ~f[i][n + 1] * ~f[j2][m] * ~f[k2][n]
						* f[i][n] * f[j2][m] * f[k2][n + 1];
			}
		}

	}
	}


//	printf("E: %f\n", Ec.real());
	atomicAdd(fval, Ec.real());
//	fval[0] = Ec.real();

//    cout << Ec.real() << endl;
//	return Ec.real();
}

double Efunc(unsigned ndim, const double *x, double *grad, void *data) {
	double* x_device;
	double* grad_device;
	device_parameters* parms_device;
	double* U_device;
	double* J_device;
	double* f_device;

	device_parameters dparms;
	parameters* parms = static_cast<parameters*>(data);

	memAlloc<double>(&x_device, ndim);
	memAlloc<double>(&grad_device, ndim);
	memAlloc<device_parameters>(&parms_device, 1);
	memAlloc<double>(&U_device, L);
	memAlloc<double>(&J_device, L);
	memAlloc<double>(&f_device, 1);

	memCopy(x_device, x, ndim * sizeof(double), cudaMemcpyHostToDevice);
	memCopy(U_device, &parms->U[0], L * sizeof(double), cudaMemcpyHostToDevice);
	memCopy(J_device, &parms->J[0], L * sizeof(double), cudaMemcpyHostToDevice);
	dparms.U = U_device;
	dparms.J = J_device;
	dparms.mu = parms->mu;
	dparms.theta = parms->theta;
	memCopy(parms_device, &dparms, sizeof(device_parameters),
		cudaMemcpyHostToDevice);

	double E = 0;

	memCopy(f_device, &E, sizeof(double), cudaMemcpyHostToDevice);
	Efuncker<<<1, L>>>(ndim, x_device, f_device, grad_device,
		parms_device);

	memCopy(grad, grad_device, ndim * sizeof(double));

	memCopy(&E, f_device, sizeof(double), cudaMemcpyDeviceToHost);

	memFree(x_device);
	memFree(grad_device);
	memFree(parms_device);
	memFree(U_device);
	memFree(J_device);
	memFree(f_device);

	cout << E << endl;

	return E;
}

__global__ void norm2sker(unsigned m, double *result, unsigned ndim,
	const double* x, double* grad, void* data) {

	const int i = threadIdx.x;
	if (i >= L) {
		return;
	}

//	const doublecomplex * f[L];
//	for (int i = 0; i < L; i++) {
//		f[i] = reinterpret_cast<const doublecomplex*>(&x[2 * i * dim]);
//	}
	const doublecomplex* f = reinterpret_cast<const doublecomplex*>(&x[2 * i
		* dim]);

//	for (int i = 0; i < L; i++) {
	double norm2i = 0;
//		result[i] = 0;
	for (int n = 0; n <= nmax; n++) {
//			result[i] += norm(f[n]);
		norm2i += norm(f[n]);
	}
//		result[i] -= 1;
	norm2i -= 1;
//	}
	result[i] = norm2i;

	if (grad) {
//		for (int i = 0; i < L; i++) {
		for (int j = 0; j < ndim; j++) {
			grad[i * ndim + j] = 2 * x[i * ndim + j];
		}
//		}
	}
}

void norm2s(unsigned m, double *result, unsigned ndim, const double* x,
	double* grad, void* data) {

	double* result_device;
	double* x_device;
	double* grad_device;

	memAlloc<double>(&result_device, L);
	memAlloc<double>(&x_device, ndim);
	memAlloc<double>(&grad_device, ndim);

	memCopy(x_device, x, ndim * sizeof(double), cudaMemcpyHostToDevice);

	norm2sker<<<1, L>>>(m, result_device, ndim, x_device, grad_device, NULL);

	memCopy(result, result_device, L * sizeof(double));
	memCopy(grad, grad_device, ndim * sizeof(double));

	memFree(x_device);
	memFree(grad_device);

//	const doublecomplex * f[L];
//	for (int i = 0; i < L; i++) {
//		f[i] = reinterpret_cast<const doublecomplex*>(&x[2 * i * dim]);
//	}
//
//	for (int i = 0; i < L; i++) {
//		result[i] = 0;
//		for (int n = 0; n <= nmax; n++) {
//			result[i] += norm(f[i][n]);
//		}
//		result[i] -= 1;
//	}
//
//	if (grad) {
//		for (int i = 0; i < L; i++) {
//			for (int j = 0; j < ndim; j++) {
//				grad[i * ndim + j] = 2 * x[i * ndim + j];
//			}
//		}
//	}
}

#endif