#ifndef _spin_h_
#define _spin_h_
#include <vector>

struct SPINinstance{
	
    double opt;
    int ell;
    std::vector<int> fvector;
	
};

double evaluateSPIN(int*, SPINinstance*);
void loadSPIN(char*, SPINinstance*);

#endif





