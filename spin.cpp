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
#include "spin.h"

using namespace std;


double evaluateSPIN(int *x, SPINinstance *inst) {
    int temp=1, counter=1;
    double result=0;
    
	for(vector<int>::iterator it = inst->fvector.begin(); it != inst->fvector.end(); it++) {
		
		if (counter%3 != 0) temp *= x[*it-1];
		
		else{
			temp *= *it;
			result += temp;
			temp=1;
		}
		
			counter++;
    }
	
	return result/(double)inst->ell;

}

void loadSPIN(char *cnf_file_name, SPINinstance *inst) {
	int temp;
	ifstream input;
	string line;
	input.open ( cnf_file_name );


	getline ( input, line );
    istringstream first( line );
    first >> inst->ell;
    
    getline ( input, line );
    istringstream second( line );
    second >> inst->opt;
    inst->opt *= -1;

    
    
	while(getline ( input, line )){
		if (line[0] == '\0') continue;
		istringstream in( line );
        while ( in >> temp ) inst->fvector.push_back(temp);

    }
        
	if ((int)inst->fvector.size() != inst->ell*6){
		cout << "Instance Error" << endl;
		exit(0);
	}

}




