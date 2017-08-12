#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "nk-wa.h"

//#define DEBUG

void loadNKWAProblem(FILE *f, NKWAProblem *problem) {
    int m;

    if (fscanf(f,"%u %u %u\n",&problem->n,&problem->k,&problem->step)!=3) {
        printf("ERROR: Could not read the header of the instance!\n");
        exit(-1);
    }

    //  printf("n=%u, k=%u, overlap=%u\n",problem->n,problem->k,problem->step);
    //  getchar();

    problem->k++;

    m=(problem->n+problem->step-1)/problem->step;

    problem->c = new double*[m];

    problem->subproblems=new int*[problem->n];
    problem->num_subproblems=new int[problem->n];

    for (int i=0; i<problem->n; i++) {
        problem->subproblems[i]=NULL;
        problem->num_subproblems[i]=0;
    }

    for (int i=0; i<m; i++) {
        int position=i*problem->step;
        int subset_size=problem->k;
        int num=1<<subset_size;

        problem->c[i] = new double[num];
        for (int j=0; j<num; j++) {
            if (fscanf(f,"%lf ",&problem->c[i][j])!=1) {
                printf("ERROR: Could not read a coefficient of the instance!\n");
                exit(-1);
            }
            //      printf("%f ",problem->c[i][j]);
        };

        for (int j=0; j<subset_size; j++) {
            problem->num_subproblems[(position+j)%problem->n]++;
#ifdef DEBUG
            //      printf("-> another subproblem for position %u\n",(position+j)%problem->n);
#endif
        }
#ifdef DEBUG
        //        getchar();
#endif
    };

    for (int i=0; i<problem->n; i++) {
        if (problem->num_subproblems[i]>0)
            problem->subproblems[i]=new int[problem->num_subproblems[i]];

        problem->num_subproblems[i]=0;
    }

    for (int i=0; i<m; i++) {
        int position=i*problem->step;
        int subset_size=problem->k;

        for (int j=0; j<subset_size; j++) {
            problem->subproblems[(position+j)%problem->n][problem->num_subproblems[(position+j)%problem->n]++]=i;
        }
    }

#ifdef DEBUG

    for (int i=0; i<problem->n; i++) {
        printf("-> position:        %u\n",i);
        printf("   num_subproblems: %u\n",problem->num_subproblems[i]);
        printf("   subproblems:     ");
        for (int j=0; j<problem->num_subproblems[i]; j++)
            printf("%u ",problem->subproblems[i][j]);
        printf("\n");
    }
    getchar();
#endif

    //  fscanf(f,"\n");
    if (fscanf(f,"%lf\n",&problem->maxF)!=1) {
        printf("ERROR: Could not read the optimum value!\n");
        exit(-1);
    }
    //  fscanf(f,"\n");
    problem->pi=new int[problem->n];
    for (int i=0; i<problem->n; i++)
        if (fscanf(f,"%d ",&problem->pi[i])!=1) {
            printf("ERROR: Cannot read the permutation!\n");
            exit(-1);
        }

    //   printf("\nmaxFitness = %f\n",problem->maxF);
    //   printf("permutation = ");
    //   for (int i=0; i<problem->n; i++)
    //     printf("%d ",problem->pi[i]);
    //   printf("\n");
};

void freeNKWAProblem(NKWAProblem *problem) {
    int m=(problem->n+problem->step-1)/problem->step;
    for (int i=0; i<m; i++)
        delete[] problem->c[i];
    for (int i=0; i<problem->n; i++)
        if (problem->subproblems[i])
            delete[] problem->subproblems[i];
    delete[] problem->subproblems;
    delete[] problem->num_subproblems;
    delete[] problem->c;
    delete[] problem->pi;
}

double evaluateNKWAProblem(char *x, NKWAProblem *problem) {
    int i,j;
    int num=0;
    double f=0;

    for (j=0; j < problem->n; j += problem->step) {
        int subset_size = problem->k;

        num=0;
        for (i=0; i < subset_size; i++)
            num = (num<<1)+x[problem->pi[(j+i)%problem->n]];

        f+=problem->c[j/problem->step][num];
    };

    return f;
}

double evaluateNKProblem(char *x, NKWAProblem *problem) {
    int i,j;
    int num=0;
    double f=0;

    for (j=0; j < problem->n; j += problem->step) {
        int subset_size = problem->k;


        if (j > problem->n - problem->k)
            subset_size = problem->n - j;


        num=0;
        for (i=0; i < subset_size; i++)
            num = (num<<1)+x[problem->pi[(j+i)%problem->n]];

        f+=problem->c[j/problem->step][num];
    };

    return f;
}

int isOptimalNKWAProblem(char *x, NKWAProblem *problem) {
    int optimal=1;
    double f=evaluateNKWAProblem(x,problem);
    if (f<problem->maxF-1E-08)
        optimal=0;

    if (f>problem->maxF+1E-10) {
        printf("ERROR: Found a better optimum!\n");
        exit(-1);
    }

    return optimal;
}

// ==============================================================================

double run_local_search_NKWA(char *x,
                             NKWAProblem *problem,
                             long *num_flips) {
    int i;
    int n=problem->n;
    double current_f;
    int num_flips_local=0;

    double max_improvement;
    current_f=evaluateNKWAProblem(x,problem);

    // !!! can still optimize but not the bottleneck so what

    do {
        max_improvement=-1;
        int change=-1;
        for (i=0; i<n; i++) {
            double improvement;

            improvement=evaluate_flip_NKWA(x,i,problem);

#ifdef DEBUG
            x[problem->pi[i]]=1-x[problem->pi[i]];
            double new_f=evaluateNKWAProblem(x,problem);
            x[problem->pi[i]]=1-x[problem->pi[i]];
            if (fabs(improvement-(new_f-current_f))>1E-10) {
                printf("Problem: Improvement: %lf (should be %lf)\n",improvement,new_f-current_f);
                getchar();
            }
            //      else
            //    {
            //      printf("Improvement computed ok (%lf=%lf)\n",improvement,new_f-current_f);
            //      getchar();
            //    }
#endif
            if (improvement>max_improvement) {
                max_improvement=improvement;
                change=problem->pi[i];
            }

        }

        if (max_improvement>0) {
            x[change]=(x[change])? 0:1;
            current_f+=max_improvement;
            num_flips_local++;
        }

        //      printf("  improvement = %e\n",max_improvement);
    } while (max_improvement>10E-15);

    (*num_flips)+=num_flips_local;

    return current_f;
}

// ==============================================================================

double evaluate_flip_NKWA(char *x, int i, NKWAProblem *problem) {
    int ii,j;
    double f=0;


    //   printf("\nEVALUATING A FLIP!\n");

    for (ii=0; ii<problem->num_subproblems[i]; ii++) {
        int which=problem->subproblems[i][ii];
        int idx=0;
        int pos_i=-1000;
        int first_pos=which*problem->step;

        int subset_size=problem->k;

        //       printf("   positions: ");
        for (j=0; j<subset_size; j++) {
            int pos=(first_pos+j)%problem->n;

            //      printf("%u(%u) ",pos,x[problem->pi[pos]]);

            idx=((idx<<1)+x[problem->pi[pos]]);

            if (pos==i)
                pos_i=j;
        }

        //       printf("\n");

        f-=problem->c[which][idx];

        /*       printf("pos_i=%u\n",pos_i); */
        /*       printf("idx before: %u\n",idx); */
        idx^=(int) (((int)1)<<(subset_size-pos_i-1));
        /*       printf("idx after:  %u (%u)\n",idx,1<<(nk->k-pos_i)); */

        /*       x[i]=1-x[i]; */
        /*       idx=0; */
        /*       for (j=0; j<=nk->k; j++) */
        /*    { */
        /*      int pos=nk->neighbors[which][j]; */

        /*      idx=((idx<<1)+x[pos]); */

        /*      if (pos==i) */
        /*        pos_i=j; */
        /*    } */
        /*       x[i]=1-x[i]; */

        /*       printf("idx correct:  %u\n",idx); */

        f+=problem->c[which][idx];
    }

    //   printf("DONE EVALUATING A FLIP!\n\n");

    return f;
}
