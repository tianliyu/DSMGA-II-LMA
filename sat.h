#ifndef _sat_h_
#define _sat_h_

struct SATinstance {

    int var;
    int clau;
    char type[30];
    std::vector<int> fvector;

};

double evaluateSAT(int*, SATinstance*);
void loadSAT(char*, SATinstance*);

#endif





