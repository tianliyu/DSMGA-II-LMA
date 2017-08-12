#ifndef _MY_GRAPH_CUH_
#define _MY_GRAPH_CUH_

__device__ __host__ inline int indexConvert(int i, int j, int ell) {
    if(i > j) {
        int tmp = i;
        i = j;
        j = tmp;
    }
    return i*(ell-1) - (i*(i+1))/2 + j - 1;
}

__device__ __host__ inline double read(double graph[], int i, int j, int ell) {
    if(i == j) {
        return 1.0;
    } else {
        return graph[indexConvert(i, j, ell)];
    }
}

__device__ inline void write(double graph[], int i, int j, double val, int ell) {
    if(i == j) {
        return;
    } else {
        graph[indexConvert(i, j, ell)] = val;
    }
}

#endif
