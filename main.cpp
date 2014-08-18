/*
 * File:   main.cpp
 * Author: Abuenameh
 *
 * Created on August 6, 2014, 11:21 PM
 */

#include <cstdlib>
#include <cstdio>
#include <math.h>
#include <nlopt.h>
#include <complex>
#include <iostream>
#include <queue>
//#include <thread>
#include <nlopt.hpp>

#include <boost/array.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/date_time.hpp>
#include <boost/random.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/progress.hpp>
#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/locks.hpp>

#include "mathematica.h"
#include "gutzwiller.hpp"

using namespace std;

//using boost::lexical_cast;
using namespace boost;
using namespace boost::random;
using namespace boost::filesystem;
using namespace boost::posix_time;

typedef boost::array<double, L> Parameter;

//template<typename T> void printMath(ostream& out, string name, T& t) {
//    out << name << "=" << ::math(t) << ";" << endl;
//}
//
//template<typename T> void printMath(ostream& out, string name, int i, T& t) {
//    out << name << "[" << i << "]" << "=" << ::math(t) << ";" << endl;
//}

boost::mutex progress_mutex;
boost::mutex points_mutex;

struct Point {
    int i;
    int j;
    double x;
    double mu;
};

#ifdef __NVCC__
#define FUNCTION extern
#else
#define FUNCTION
#endif

double Efunc(unsigned ndim, const double *x, double *grad, void *data) {
    parameters* parms = static_cast<parameters*> (data);
    if (parms->canonical) {
        return Ecfunc(ndim, x, grad, data);
    }
    else {
        return Encfunc(ndim, x, grad, data);
    }
}

//double Efunc(unsigned ndim, const double *x, double *grad, void *data);
//double Ectfunc(unsigned ndim, const double *x, double *grad, void *data);
FUNCTION void norm2s(unsigned m, double *result, unsigned ndim, const double* x,
        double* grad, void* data);

double norm2(const vector<double> x, vector<double>& norm2is) {
    const doublecomplex * f[L];
    for (int i = 0; i < L; i++) {
        f[i] = reinterpret_cast<const doublecomplex*> (&x[2 * i * dim]);
    }

    norm2is.resize(L);

    double norm2 = 1;
    for (int i = 0; i < L; i++) {
        double norm2i = 0;
        for (int n = 0; n <= nmax; n++) {
            norm2i += norm(f[i][n]);
        }
        norm2 *= norm2i;
        norm2is[i] = norm2;
    }
    return norm2;

}

int min(int a, int b) {
    return a < b ? a : b;
}

int max(int a, int b) {
    return a > b ? a : b;
}

void phasepoints(Parameter& xi, phase_parameters pparms, queue<Point>& points, multi_array<vector<double>, 2 >& f0, multi_array<double, 2 >& E0res, multi_array<double, 2 >& Ethres, multi_array<double, 2 >& Eth2res, multi_array<double, 2 >& fs, progress_display& progress) {

    int ndim = 2 * L * dim;

    vector<double> x(ndim);
    vector<double> U(L), J(L);

    parameters parms;
    parms.canonical = pparms.canonical;
    //        parms.theta = theta;
    
    double theta = pparms.theta;

    for (;;) {
        Point point;
        {
            boost::mutex::scoped_lock lock(points_mutex);
            if (points.empty()) {
                break;
            }
            point = points.front();
            points.pop();
        }
        //        cout << "Got queued" << endl;

        //
        //    vector<double> U(L), J(L);
        for (int i = 0; i < L; i++) {
            ////		U[i] = 0.1 * sqrt(i + 1);
            ////		J[i] = 0.1 * min(i + 1, mod(i + 1) + 1)
            ////			+ 0.2 * max(i + 1, mod(i + 1) + 1);
            U[i] = 1;
            J[i] = point.x;
        }

        doublecomplex * f[L];
        for (int i = 0; i < L; i++) {
            f[i] = reinterpret_cast<doublecomplex*> (&x[2 * i * dim]);
        }

        for (int i = 0; i < L; i++) {
            for (int n = 0; n <= nmax; n++) {
                f[i][n] = 1 / sqrt(dim);
            }
        }
        //    
        //	parameters parms;
        parms.J = J;
        parms.U = U;
        parms.mu = point.mu;
        
        parms.theta = 0;
        //
        ////    Efuncth(ndim, &x[0], NULL, &parms);
        ////    return 0;
        //
        //        cout << "Setting up optimizer" << endl;
        nlopt::opt opt(nlopt::LD_LBFGS, ndim);
        opt.set_lower_bounds(-1);
        opt.set_upper_bounds(1);
        vector<double> ctol(L, 1e-8);
        //        opt.add_equality_mconstraint(norm2s, NULL, ctol);
        opt.set_xtol_rel(1e-8);

        opt.set_min_objective(Efunc, &parms);
        //            cout << "Optimizer set up. Doing optimization" << endl;

        int res = 0;

        double E0 = 0;
        try {
            res = opt.optimize(x, E0);
            //            printf("Ground state energy: %0.10g\n", E0);
        }        catch (std::exception& e) {
            printf("nlopt failed! %d\n", res);
            cout << e.what() << endl;
            E0 = numeric_limits<double>::quiet_NaN();
        }

        f0[point.i][point.j] = x;
        E0res[point.i][point.j] = E0;

        //        opt.set_min_objective(Ethfunc, &parms);

        double Eth = 0;
        parms.theta = theta;
        try {
            res = opt.optimize(x, Eth);
            //            printf("Twisted energy: %0.10g\n", Eth);
        }        catch (std::exception& e) {
            printf("nlopt failed! %d\n", res);
            cout << e.what() << endl;
            Eth = numeric_limits<double>::quiet_NaN();
        }

        Ethres[point.i][point.j] = Eth;

        double Eth2 = 0;
        parms.theta = 2 * theta;
        try {
            res = opt.optimize(x, Eth2);
            //            printf("Twisted energy 2: %0.10g\n", Eth2);
        }        catch (std::exception& e) {
            printf("nlopt failed! %d\n", res);
            cout << e.what() << endl;
            Eth2 = numeric_limits<double>::quiet_NaN();
        }

        Eth2res[point.i][point.j] = Eth2;

        fs[point.i][point.j] = (Eth2 - 2 * Eth + E0) / (theta * theta);
        //        cout << "fs = " << (Eth2-2*Eth+E0)/(0.01*0.01) << endl;

        //    
        //        cout << "Eth - E0 = " << Eth-E0 << endl << endl;

        {
            boost::mutex::scoped_lock lock(progress_mutex);
            ++progress;
        }
    }

}

/*
 *
 */
int main(int argc, char** argv) {

    //    vector<double> f(2*L*dim,1/sqrt(2.*dim));
    //    vector<double> g(2*L*dim,0);
    //    	parameters parms;
    //    	parms.J = vector<double>(L,0.1);
    //    	parms.U = vector<double>(L,1);
    //    	parms.mu = 0.5;
    //        parms.theta = 0;
    //    double E1 = Ectfunc(2*L*dim,f.data(),g.data(),&parms);
    //    int id = 0;
    //    f[id] += 1e-6;
    //    double E2 = Ectfunc(2*L*dim,f.data(),g.data(),&parms);
    //    cout << g[id] << endl;
    //    cout << (E2-E1) << endl;
    //    
    //    return 0;

    mt19937 rng;
    uniform_real_distribution<> uni(-1, 1);

    int seed = lexical_cast<int>(argv[1]);
    int nseed = lexical_cast<int>(argv[2]);

    double xmin = lexical_cast<double>(argv[3]);
    double xmax = lexical_cast<double>(argv[4]);
    int nx = lexical_cast<int>(argv[5]);

    deque<double> x(nx);
    if (nx == 1) {
        x[0] = xmin;
    } else {
        double dx = (xmax - xmin) / (nx - 1);
        for (int ix = 0; ix < nx; ix++) {
            x[ix] = xmin + ix * dx;
        }
    }

    double mumin = lexical_cast<double>(argv[6]);
    double mumax = lexical_cast<double>(argv[7]);
    int nmu = lexical_cast<int>(argv[8]);

    deque<double> mu(nmu);
    if (nmu == 1) {
        mu[0] = mumin;
    } else {
        double dmu = (mumax - mumin) / (nmu - 1);
        for (int imu = 0; imu < nmu; imu++) {
            mu[imu] = mumin + imu * dmu;
        }
    }

    double D = lexical_cast<double>(argv[9]);
    double theta = lexical_cast<double>(argv[10]);

    int numthreads = lexical_cast<int>(argv[11]);

    int resi = lexical_cast<int>(argv[12]);
    
    bool canonical = lexical_cast<bool>(argv[13]);

#ifdef AMAZON
    //    path resdir("/home/ubuntu/Dropbox/Amazon EC2/Simulation Results/Gutzwiller Phase Diagram");
    path resdir("/home/ubuntu/Dropbox/Amazon EC2/Simulation Results/Canonical Transformation Gutzwiller");
#else
    //    path resdir("/Users/Abuenameh/Dropbox/Amazon EC2/Simulation Results/Gutzwiller Phase Diagram");
    path resdir("/Users/Abuenameh/Documents/Simulation Results/Canonical Transformation Gutzwiller");
#endif
    if (!exists(resdir)) {
        cerr << "Results directory " << resdir << " does not exist!" << endl;
        exit(1);
    }
    for (int iseed = 0; iseed < nseed; iseed++, seed++) {
        ptime begin = microsec_clock::local_time();


        ostringstream oss;
        oss << "res." << resi << ".txt";
        path resfile = resdir / oss.str();
        while (exists(resfile)) {
            resi++;
            oss.str("");
            oss << "res." << resi << ".txt";
            resfile = resdir / oss.str();
        }
        if (seed < 0) {
            resi = seed;
            oss.str("");
            oss << "res." << resi << ".txt";
            resfile = resdir / oss.str();
        }

        Parameter xi;
        xi.fill(1);
        //        xi.assign(1);
        rng.seed(seed);
        if (seed > -1) {
            for (int j = 0; j < L; j++) {
                xi[j] = (1 + D * uni(rng));
            }
        }

        int Lres = L;

        boost::filesystem::ofstream os(resfile);
        printMath(os, "Lres", resi, Lres);
        printMath(os, "seed", resi, seed);
        printMath(os, "theta", resi, theta);
        printMath(os, "Delta", resi, D);
        printMath(os, "xres", resi, x);
        printMath(os, "mures", resi, mu);
        printMath(os, "xires", resi, xi);
        os << flush;

        cout << "Res: " << resi << endl;

        //        multi_array<double, 2 > fcres(extents[nx][nmu]);
        multi_array<double, 2 > fsres(extents[nx][nmu]);
        //        multi_array<double, 2> dur(extents[nx][nmu]);
        //        multi_array<int, 2> iterres(extents[nx][nmu]);
        multi_array<vector<double>, 2> f0res(extents[nx][nmu]);
        multi_array<double, 2> E0res(extents[nx][nmu]);
        multi_array<double, 2> Ethres(extents[nx][nmu]);
        multi_array<double, 2> Eth2res(extents[nx][nmu]);

        progress_display progress(nx * nmu);

        //        cout << "Queueing" << endl;
        queue<Point> points;
        for (int imu = 0; imu < nmu; imu++) {
            queue<Point> rowpoints;
            for (int ix = 0; ix < nx; ix++) {
                Point point;
                point.i = ix;
                point.j = imu;
                point.x = x[ix];
                point.mu = mu[imu];
                points.push(point);
            }
        }
        
        phase_parameters parms;
        parms.theta = theta;
        parms.canonical = canonical;

        //        cout << "Dispatching" << endl;
        thread_group threads;
        //        vector<thread> threads;
        for (int i = 0; i < numthreads; i++) {
            //                        threads.emplace_back(phasepoints, std::ref(xi), theta, std::ref(points), std::ref(f0res), std::ref(E0res), std::ref(Ethres), std::ref(fsres), std::ref(progress));
            threads.create_thread(bind(&phasepoints, boost::ref(xi), parms, boost::ref(points), boost::ref(f0res), boost::ref(E0res), boost::ref(Ethres), boost::ref(Eth2res), boost::ref(fsres), boost::ref(progress)));
        }
        //        for (thread& t : threads) {
        //            t.join();
        //        }
        threads.join_all();


        //        printMath(os, "fcres", resi, fcres);
        printMath(os, "fsres", resi, fsres);
        //                printMath(os, "dur", resi, dur);
        //        printMath(os, "iters", resi, iterres);
        printMath(os, "f0res", resi, f0res);
        printMath(os, "E0res", resi, E0res);
        printMath(os, "Ethres", resi, Ethres);
        printMath(os, "Eth2res", resi, Eth2res);

        ptime end = microsec_clock::local_time();
        time_period period(begin, end);
        cout << endl << period.length() << endl << endl;

        os << "runtime[" << resi << "]=\"" << period.length() << "\";" << endl;
    }

    //    time_t start = time(NULL);
    //
    //    int ndim = 2 * L * dim;

    //    vector<double> x(ndim);
    //
    //    vector<double> U(L), J(L);
    //	for (int i = 0; i < L; i++) {
    ////		U[i] = 0.1 * sqrt(i + 1);
    ////		J[i] = 0.1 * min(i + 1, mod(i + 1) + 1)
    ////			+ 0.2 * max(i + 1, mod(i + 1) + 1);
    //        U[i] = 1;
    //        J[i] = 0.2;
    //	}
    //
    //	doublecomplex * f[L];
    //	for (int i = 0; i < L; i++) {
    //		f[i] = reinterpret_cast<doublecomplex*>(&x[2 * i * dim]);
    //	}
    //
    //	for (int i = 0; i < L; i++) {
    //		for (int n = 0; n <= nmax; n++) {
    //            f[i][n] = 1/sqrt(dim);
    //		}
    //	}
    //    
    //	parameters parms;
    //	parms.J = J;
    //	parms.U = U;
    //	parms.mu = 0.5;
    //    parms.theta = 0.1;
    //
    ////    Efuncth(ndim, &x[0], NULL, &parms);
    ////    return 0;
    //
    //    nlopt::opt opt(/*nlopt::GN_ISRES*/nlopt::LN_COBYLA/*nlopt::LD_SLSQP*/, ndim);
    ////    nlopt::opt opt(nlopt::AUGLAG/*nlopt::GN_ISRES*//*nlopt::LN_COBYLA*//*nlopt::LD_SLSQP*/, ndim);
    ////    nlopt::opt local_opt(nlopt::LN_SBPLX, ndim);
    ////    opt.set_local_optimizer(local_opt);
    //    opt.set_lower_bounds(-1);
    //    opt.set_upper_bounds(1);
    //    vector<double> ctol(L, 1e-8);
    //    opt.add_equality_mconstraint(norm2s, NULL, ctol);
    //    opt.set_min_objective(Efunc, &parms);
    //    opt.set_xtol_rel(1e-8);
    //
    //	int res = 0;
    //    
    //    double E0 = 0;
    //    try {
    //        res = opt.optimize(x, E0);
    //        printf("Found minimum: %0.10g\n", E0);
    //    }
    //    catch(exception& e) {
    //        printf("nlopt failed! %d\n", res);
    //        cout << e.what() << endl;
    //    }
    //    
    //    opt.set_min_objective(Efuncth, &parms);
    //    
    //    double Eth = 0;
    //    try {
    //        res = opt.optimize(x, Eth);
    //        printf("Found minimum: %0.10g\n", Eth);
    //    }
    //    catch(exception& e) {
    //        printf("nlopt failed! %d\n", res);
    //        cout << e.what() << endl;
    //    }
    //    
    //    cout << "Eth - E0 = " << Eth-E0 << endl << endl;
    //
    //    for(int i = 0; i < 1; i++) {
    //        for(int n = 0; n <= nmax; n++) {
    //        cout << norm(f[i][n]) << endl;
    //        }
    //        cout << endl;
    //    }

    //    vector<double> norm2is;
    //	cout << norm2(x, norm2is) << endl;
    //	cout << E0 / norm2(x, norm2is) << endl;

    //    nlopt_set_xtol_rel(opt, 1e-8);
    //
    //    double x[2] = {1.234, 5.678}; /* some initial guess */
    //    double minf; /* the minimum objective value, upon return */
    //
    //    int res = 0;
    //    if ((res = nlopt_optimize(opt, x, &minf)) < 0) {
    //        printf("nlopt failed! %d\n",res);
    //    } else {
    //        printf("found minimum at f(%g,%g) = %0.10g\n", x[0], x[1], minf);
    //    }
    //
    //nlopt_destroy(opt);


    //    time_t end = time(NULL);
    //
    //    printf("Runtime: %ld", end - start);

    return 0;
}

