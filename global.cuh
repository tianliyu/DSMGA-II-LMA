/***************************************************************************
 *   Copyright (C) 2015 Tian-Li Yu and Shih-Huan Hsu                       *
 *   tianliyu@ntu.edu.tw                                                   *
 ***************************************************************************/


#ifndef _GLOBAL_CUH
#define _GLOBAL_CUH

#define NDEBUG // for assert
#include <cassert>
#include <cmath>

#include "myrand.h"
#include "bitwisedistance.h"
#include "spin.h"
#include "nk-wa.h"
#include "doublelinkedlistarray.h"
#include "zkey.h"
#include "sat.h"
#include "curand_kernel.h"

#define EPSILON (1e-8)
#define INF (1e10)

extern bool GHC;
extern bool SELECTION;
extern bool CACHE;
extern bool SHOW_BISECTION;

extern char outputFilename[100];
extern void gstop ();
extern void outputErrMsg (const char *errMsg);
extern int pow2 (int x);

extern curandState_t *d_randStates;
extern ZKey zKey;
extern MyRand myRand;
extern BitwiseDistance myBD;
extern SPINinstance mySpinGlassParams;
extern SATinstance mySAT;

extern NKWAProblem nkwa;

inline int quotientLong(int a) {
    return (a / (sizeof(unsigned long) * 8) );
}

inline int remainderLong(int a) {
    return (a & (sizeof(unsigned long) * 8 - 1));
}

inline double jointEntropy(double p00, double p01, double p10, double p11) {
    double result = 0.0;
    result -= p00 * log(p00);
    result -= p01 * log(p01);
    result -= p10 * log(p10);
    result -= p11 * log(p11);

    return result;
}

inline double mutualInformation(double p00, double p01, double p10, double p11) {
    double result = 0.0;

    double p0x = p00+p01;
    double p1x = p10+p11;
    double px0 = p00+p10;
    double px1 = p01+p11;

    result += (p00 < EPSILON) ? 0.0 : p00 * log (p00 / p0x / px0);
    result += (p01 < EPSILON) ? 0.0 : p01 * log (p01 / p0x / px1);
    result += (p10 < EPSILON) ? 0.0 : p10 * log (p10 / p1x / px0);
    result += (p11 < EPSILON) ? 0.0 : p11 * log (p11 / p1x / px1);

    return result;
}

inline double metric(double p00, double p01, double p10, double p11) {
    return mutualInformation(p00,p01,p10,p11)/jointEntropy(p00,p01,p10,p11);
}

inline double square(double a) {
    return a*a;
}

#endif
