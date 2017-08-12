/***************************************************************************
 myrand.cc  -  description
 -------------------
 begin                : Sep 24 2001
 copyright            : (C) 2001 by Tian-Li Yu
 email                : tianliyu@illigal.ge.uiuc.edu
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include <math.h>
#include <time.h>
#include <stdlib.h>
#include "myrand.h"

#ifdef PI
#undef PI
#endif

#define PI 3.14159265
#define N 624

MyRand::MyRand() {

    unsigned long init_key[N];

    srand((unsigned long) time(NULL));

    for (int i = 0; i < N; i++) {
        init_key[i] = (unsigned long) time(NULL) * rand();
    }

    init_by_array(init_key, N);
}

MyRand::~MyRand() {
}

void MyRand::seed(unsigned long seed) {
    init_genrand(seed);
}

bool MyRand::flip() {
    static unsigned long mask = (1ul << 31);
    static unsigned long x = genrand_int32();

    bool result = ((mask & x) == 0);

    mask >>= 1;

    if (mask == 0) {
        mask = (1ul << 31);
        x = genrand_int32();
    }

    return result;
}

bool MyRand::flip(double prob) {
    return (uniform() < prob);
}

/** From [0,1] */
double MyRand::uniform() {
    return genrand_real1();
}

/** From [a,b] */
double MyRand::uniform(double a, double b) {
    return uniform() * (b - a) + a;
}

/** Standard normal distribution */
double MyRand::normal() {

    double u1, u2, z;
    u1 = genrand_real3(); // (0,1)
    u2 = uniform();

    z = sqrt(-2 * log(u1)) * sin(2 * PI * u2);

    return z;
}

/** Normal distribution with mean and standard deviation */
double MyRand::normal(double mean, double stdd) {
    return normal() * stdd + mean;
}

int MyRand::uniformInt(int a, int b) {
    return (a + (int) (genrand_real2() * (b - a + 1)));
}

void MyRand::uniformArray(int *array, int num, int a, int b) {

    int *base = new int[b - a + 1];
    int i;
    int r;

    for (i = 0; i < b - a + 1; i++)
        base[i] = a + i;

    for (i = 0; i < num; i++) {
        r = uniformInt(0, b - a - i);
        array[i] = base[r];
        base[r] = base[b - a - i];
    }

    delete[] base;
}

int MyRand::dice(double *pr, int size, double prSum) {
    // if we don't know prSum
    if (prSum < 0.0) {
        prSum = 0.0;
        for (int i = 0; i < size; i++)
            prSum += pr[i];
    }

    double x = uniform(0, prSum);
    double partial = 0.0;

    for (int i = 0; i < size; i++) {
        if (partial > x)
            return i;
        partial += pr[i];
    }

    return size - 1;
}
