/***************************************************************************
 *   Copyright (C) 2006 by tianliyu                                        *
 *   tianliyu@illigal.ge.uiuc.edu                                          *
 *                                                                         *
 ***************************************************************************/
#ifndef _FASTCOUNTING_H
#define _FASTCOUNTING_H

/**
@author Tian-Li Yu
*/

class FastCounting {

public:
    FastCounting ();
    FastCounting (int n_ell);

    ~FastCounting ();

    void init (int n_ell);

    int getVal (int index) const;
    void setVal (int index, int val);

    FastCounting& operator= (const FastCounting& c);

    int makeInt (int *bb) const;

    int getLength () const;
    void setLength ();


    unsigned int getInt(int start, int length) const;

    unsigned long *gene;
    int length;
    int lengthLong;

};

#endif
