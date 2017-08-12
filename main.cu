/***************************************************************************
 *   Copyright (C) 2015 Tian-Li Yu and Shih-Huan Hsu                       *
 *   tianliyu@ntu.edu.tw                                                   *
 ***************************************************************************/


#include <math.h>
#include <iostream>
#include <cstdlib>

#include "statistics.h"
#include "dsmga2.cuh"
#include "global.cuh"
#include "chromosome.h"

using namespace std;

__global__ void initCurandStates(curandState_t *d_randStates, int randSeed);

int
main (int argc, char *argv[]) {


    if (argc < 9) {
        printf ("DSMGA2 ell nInitial function maxGen maxFe repeat display rand_seed\n");
        printf ("function: \n");
        printf ("     ONEMAX:  0\n");
        printf ("     MK    :  1\n");
        printf ("     FTRAP :  2\n");
        printf ("     CYC   :  3\n");
        printf ("     NK    :  4\n");
        printf ("     SPIN  :  5\n");
        printf ("     SAT   :  6\n");

        return -1;
    }

    int ell = atoi (argv[1]); // problem size
    int nInitial = atoi (argv[2]); // initial population size
    int fffff = atoi (argv[3]); // function
    int maxGen = atoi (argv[4]); // max generation
    int maxFe = atoi (argv[5]); // max fe
    int repeat = atoi (argv[6]); // how many time to repeat
    int display = atoi (argv[7]); // display each generation or not
    int rand_seed = atoi (argv[8]);  // rand seed


    if (fffff == 4) {

        char filename[200];
        sprintf(filename, "./NK_Instance/pnk%d_%d_%d_%d", ell, 4, 1, 1);

        if (SHOW_BISECTION) printf("Loading: %s\n", filename);
        FILE *fp = fopen(filename, "r");
        loadNKWAProblem(fp, &nkwa);
        fclose(fp);
    }

    if (fffff == 5) {
        int instance = atoi(argv[9]);
        char filename[200];
        sprintf(filename, "./SPIN/%d/%d_%d",ell, ell, instance);
        if (SHOW_BISECTION) printf("Loading: %s\n", filename);
        loadSPIN(filename, &mySpinGlassParams);
    }

    if (fffff == 6) {
        char filename[200];
        sprintf(filename, "./SAT/uf%d/uf%d-0%d.cnf", ell, ell, 1);
        if (SHOW_BISECTION) printf("Loading: %s\n", filename);
        loadSAT(filename, &mySAT);
    }


    if (rand_seed != -1)  // time
        myRand.seed((unsigned long)rand_seed);

    int i;

    Statistics stGen, stFE, stLSFE, stBuildModelTime, stFindCliqueTime;
    int usedGen;

    int failNum = 0;

    cudaMalloc(&d_randStates, sizeof(curandState_t) * ell * ell);
    initCurandStates<<<ell, ell>>>(d_randStates, rand_seed);

    for (i = 0; i < repeat; i++) {

        DSMGA2 ga (ell, nInitial, maxGen, maxFe, fffff, rand_seed+i);

        if (display == 1)
            usedGen = ga.doIt (true);
        else
            usedGen = ga.doIt (false);


        if (!ga.foundOptima()) {
            failNum++;
            printf ("-");
        } else {
            stFE.record (Chromosome::hitnfe);
            stLSFE.record (Chromosome::lsnfe);
            stGen.record (usedGen);
            stBuildModelTime.record(ga.buildModelTime);
            stFindCliqueTime.record(ga.findCliqueTime);
            printf ("+");
        }

        fflush (NULL);

    }

    cout << endl;
    printf ("\n");
    printf ("%f  %f  %f %d\n", stGen.getMean (), stFE.getMean(), stLSFE.getMean(), failNum);
    printf("build model time: %f   find clique time: %f\n", stBuildModelTime.getMean(), stFindCliqueTime.getMean());

    if (fffff == 4) freeNKWAProblem(&nkwa);
    cudaFree(d_randStates);

    return EXIT_SUCCESS;
}

__global__ void initCurandStates(curandState_t *d_randStates, int randSeed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(randSeed, idx, 0, &d_randStates[idx]);
}
