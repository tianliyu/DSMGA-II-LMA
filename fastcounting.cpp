/***************************************************************************
 *   Copyright (C) 2004 by tianliyu                                        *
 *   tianliyu@illigal.ge.uiuc.edu                                          *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/

#include <cstring>
#include "global.cuh"
#include "fastcounting.h"


FastCounting::FastCounting () {
    length = 0;
    lengthLong = 0;
    gene = NULL;
}

FastCounting::FastCounting (int n_length) {
    gene = NULL;

    init (n_length);
}


FastCounting::~FastCounting () {
    if (gene != NULL) delete []gene;
}

void FastCounting::init (int n_length) {
    length = n_length;
    lengthLong = quotientLong(length)+1;

    if (gene != NULL)
        delete []gene;

    gene = new unsigned long [lengthLong];
    gene[lengthLong-1] = 0;

}

unsigned int FastCounting::getInt (int start, int length) const {
    assert (length < (int) sizeof(unsigned int) * 8);

    int q = quotientLong(start);
    int r = remainderLong(start);

    if ( r + length <= (int) sizeof(unsigned long) * 8) {
        // with in one long
        unsigned long mask = (~(0lu)) >> (sizeof(unsigned long) * 8 - length);
        mask <<= r;

        return (unsigned int) ((gene[q] & mask) >> r);
    } else {
        unsigned long mask = (~(0lu)) << r;
        int part1 = ((gene[q] & mask) >> r);

        int length2 = (r + length) - sizeof(unsigned long) * 8;
        int length1 = length - length2;

        mask = (~(0lu)) >> (sizeof(unsigned long) * 8 - length2);
        int part2 = (gene[q+1] & mask);

        return ((part2 << length1) & part1);
    }

}

int FastCounting::getVal (int index) const {
    assert (index >= 0 && index < length);

    int q = quotientLong(index);
    int r = remainderLong(index);

    if ( (gene[q] & (1lu << r)) == 0 )
        return 0;
    else
        return 1;
}

void FastCounting::setVal (int index, int val) {
    assert (index >= 0 && index < length);


    int q = quotientLong(index);
    int r = remainderLong(index);

    if (val == 1)
        gene[q] |= (1lu<<r);
    else
        gene[q] &= ~(1lu<<r);

}

FastCounting& FastCounting::operator= (const FastCounting& c) {

    if (length != c.length) {
        length = c.length;
        init (length);
    }

    lengthLong = c.lengthLong;

    memcpy(gene, c.gene, sizeof(long) * lengthLong);

    return *this;
}

int FastCounting::makeInt (int *bb) const {
    int value = 0;

    for (int i = 1; i <= bb[0]; i++)
        if (getVal(bb[i]) == 1)
            value |= (1 << (i - 1));

    return value;
}

int FastCounting::getLength () const {
    return length;
}


