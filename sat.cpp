/***************************************************************************
 *   Copyright (C) 2015 Tian-Li Yu and Shih-Huan Hsu                       *
 *   tianliyu@ntu.edu.tw                                                   *
 ***************************************************************************/

#include <iostream>
#include <iomanip>
#include <cstring>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "sat.h"

using namespace std;


double evaluateSAT(int *x,  SATinstance *inst) {

    double f=0;
    bool b=false;

    for(vector<int>::iterator it = inst->fvector.begin(); it != inst->fvector.end(); it++) {

        if ( b && *it!=0 ) continue;

        if (*it != 0) {

            if (*it>0) {
                if (x[(*it)-1] == 1)
                    b = true;
            } else {
                if (x[-1*(*it+1)]==0)
                    b = true;
            }
        } else {
            if (!b)
                f -= 1;
            b = false;
        }
    }
    return f;

}

void loadSAT(char *cnf_file_name, SATinstance *inst) {
    int temp;
    ifstream input;
    string line;
    input.open ( cnf_file_name );

    if(!input) {
        cout << "Fail to open file: " << cnf_file_name << endl;
        exit(0);
    }


    getline ( input, line );

    while (line[0] != 'p' || line[0] == 'P') {
        getline ( input, line );
    }

    istringstream in( line );
    in >> inst->type;
    in >> inst->type;
    in >> inst->var;
    in >> inst->clau;


    while(getline ( input, line )) {
        if (line[0] == '%') break;
        istringstream in( line );
        while ( in >> temp ) inst->fvector.push_back(temp);

    }
    input.close();

}




