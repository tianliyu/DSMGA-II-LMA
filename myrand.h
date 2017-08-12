/***************************************************************************
                          myrand.h  -  description
                             -------------------
    begin                : Aug 24 2001
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

#ifndef MYRAND_H
#define MYRAND_H


/**
  *@author Tian-Li Yu
  */

#include "mt19937ar.h"


class MyRand {
public:
    MyRand ();
    ~MyRand ();

    void seed(unsigned long);

    bool flip();
    bool flip(double prob);

    /** From [a,b) */
    double uniform(double a, double b);

    /** From [0,1) */
    double uniform();

    /** Standard normal distribution */
    double normal();
    /** Normal distribution with mean and standard deviation */
    double normal(double mean, double std);
    /** No descriptions */
    int uniformInt(int a, int b);
    /** No descriptions */
    void uniformArray(int *array, int num, int a, int b);

    /** dice according to pr */
    int dice(double *pr, int size, double prSum=-1.0);
};


#endif
