#ifndef _SEPARABLE_NK_WA_H_
#define _SEPARABLE_NK_WA_H_

#include  <stdio.h>

typedef struct {
    int n;
    int k;
    int step;

    double **c;
    int *pi;

    double maxF;

    int *num_subproblems;
    int **subproblems;
} NKWAProblem;

void loadNKWAProblem(FILE *f,
                     NKWAProblem *problem);

void freeNKWAProblem(NKWAProblem *problem);

double evaluateNKWAProblem(char *x,
                           NKWAProblem *problem);

double evaluateNKProblem(char *x,
                         NKWAProblem *problem);

int isOptimalNKWAProblem(char *x,
                         NKWAProblem *problem);

int isOptimalShuffledNKWAProblem(char *x,
                                 NKWAProblem *problem);

double run_local_search_NKWA(char *x,
                             NKWAProblem *problem,
                             long *num_flips);

double evaluate_flip_NKWA(char *x,
                          int i,
                          NKWAProblem *problem);

#endif
