
#ifndef _BITWISEDISTANCE_H_
#define _BITWISEDISTANCE_H_

class BitwiseDistance {

public:

    int countOne(unsigned long bitString) {
        return __builtin_popcountl(bitString);
    }
    int getHammingDistance(unsigned long a, unsigned long b) {
        return countOne(a^b);
    }

};


#endif
