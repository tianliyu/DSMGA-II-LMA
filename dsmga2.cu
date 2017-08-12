/***************************************************************************
 *   Copyright (C) 2015 Tian-Li Yu and Shih-Huan Hsu                       *
 *   tianliyu@ntu.edu.tw                                                   *
 ***************************************************************************/

#include <list>
#include <vector>
#include <algorithm>
#include <iterator>

#include <iostream>
#include <stdio.h>
#include "chromosome.h"
#include "dsmga2.cuh"
#include "statistics.h"
#include "mygraph.cuh"

using namespace std;

__device__ __host__ inline int quotientInt(int a) {
    return (a / (sizeof(unsigned int) * 8));
}

__device__ __host__ inline int remainderInt(int a) {
    return (a & (sizeof(unsigned int) * 8 - 1));
}
__global__ void deviceBuildFastCounting(unsigned int *d_totalGene, unsigned int *d_fastCounting, int *d_selectionPool, int chGeneLength, int fcGeneLength, size_t tgPitch, size_t fcPitch, int ell, int nCurrent);
__global__ void deviceBuildGraph(double*d_myGraph, unsigned int *d_fastCounting, size_t fcPitch, int fcGeneLength, int ell, int nCurrent, int graphSize);
__global__ void deviceFindClique(double *d_myGraph, int *d_masks, unsigned int *d_totalGene, int chGeneLength, size_t tgPitch, int ell, int nCurrent, curandState_t *d_randStates);
__device__ __host__ int getVal(unsigned int *gene, int idx);
__device__ void setVal(unsigned int *gene, int idx, int val);
__device__ int countOnes(unsigned int *gene, int fcGeneLength);
__device__ int countXOR(unsigned int *gene1, unsigned int *gene2, int fcGeneLength);
__device__ double deviceComputeMI(double a00, double a01, double a10, double a11);

DSMGA2::DSMGA2 (int n_ell, int n_nInitial, int n_maxGen, int n_maxFe, int fffff, int randSeed) {
    cudaEventCreate(&buildModelStart);
    cudaEventCreate(&buildModelStop);
    cudaEventCreate(&findCliqueStart);
    cudaEventCreate(&findCliqueStop);
    buildModelTime = 0;
    findCliqueTime = 0;

    previousFitnessMean = -INF;
    ell = n_ell;
    nCurrent = (n_nInitial/2)*2;  // has to be even

    Chromosome::function = (Chromosome::Function)fffff;
    Chromosome::nfe = 0;
    Chromosome::lsnfe = 0;
    Chromosome::hitnfe = 0;
    Chromosome::hit = false;

    selectionPressure = 2;
    maxGen = n_maxGen;
    maxFe = n_maxFe;

    bestIndex = -1;
    masks = new list<int>[ell];
    selectionIndex = new int[nCurrent];
    orderN = new int[nCurrent];
    orderELL = new int[ell];
    tmp_masks = new int[ell * ell];
    population = new Chromosome[nCurrent];

    // GPU
    selectionPool = new int[selectionPressure * nCurrent];
    cudaMalloc(&d_selectionPool, sizeof(int) * selectionPressure * nCurrent);
    fcGeneLength = quotientInt(nCurrent) + 1;
    chGeneLength = quotientInt(ell) + 1;
    totalGene = new unsigned int[nCurrent * chGeneLength];
    cudaMallocPitch(&d_fastCounting, &fcPitch, sizeof(unsigned int) * fcGeneLength, ell);
    cudaMallocPitch(&d_totalGene, &tgPitch, sizeof(int) * chGeneLength, nCurrent);
    graphSize = ell*(ell-1)/2;
    myGraph = new double[graphSize];
    cudaMalloc(&d_myGraph, sizeof(double) * graphSize);
    cudaMalloc(&d_masks, sizeof(int) * ell * ell);

    pHash.clear();
    for (int i=0; i<nCurrent; ++i) {
        population[i].initR(ell);
        double f = population[i].getFitness();
        pHash[population[i].getKey()] = f;
    }

    if (GHC) {
        for (int i=0; i < nCurrent; i++) {
            population[i].GHC();
        }
    }
}


DSMGA2::~DSMGA2 () {
    delete []masks;
    delete []orderN;
    delete []orderELL;
    delete []selectionIndex;
    delete []population;
    delete []selectionPool;
    delete []totalGene;
    delete []myGraph;
    delete []tmp_masks;
    cudaFree(d_fastCounting);
    cudaFree(d_totalGene);
    cudaFree(d_selectionPool);
    cudaFree(d_myGraph);
    cudaFree(d_masks);
}



bool DSMGA2::isSteadyState () {
    if (stFitness.getNumber () <= 0) {
        return false;
    }

    if (previousFitnessMean < stFitness.getMean ()) {
        previousFitnessMean = stFitness.getMean () + EPSILON;
        return false;
    }

    return true;
}



int DSMGA2::doIt (bool output) {
    generation = 0;

    while (!shouldTerminate ()) {
        oneRun (output);
    }
    return generation;
}


void DSMGA2::oneRun (bool output) {
    if (CACHE) {
        Chromosome::cache.clear();
    }

    mixing();

    double max = -INF;
    stFitness.reset ();

    for (int i = 0; i < nCurrent; ++i) {
        double fitness = population[i].getFitness();
        if (fitness > max) {
            max = fitness;
            bestIndex = i;
        }
        stFitness.record (fitness);
    }

    if (output) {
        showStatistics ();
    }

    ++generation;
}


bool DSMGA2::shouldTerminate () {
    bool
    termination = false;

    if (maxFe != -1) {
        if (Chromosome::nfe > maxFe) {
            termination = true;
        }
    }

    if (maxGen != -1) {
        if (generation > maxGen) {
            termination = true;
        }
    }

    if (population[0].getMaxFitness() <= stFitness.getMax()) {
        termination = true;
    }

    if (stFitness.getMax() - EPSILON <= stFitness.getMean()) {
        termination = true;
    }

    return termination;

}


bool DSMGA2::foundOptima () {
    return (stFitness.getMax() > population[0].getMaxFitness());
}


void DSMGA2::showStatistics () {
    printf ("Gen:%d  Fitness:(Max/Mean/Min):%f/%f/%f \n ",
            generation, stFitness.getMax (), stFitness.getMean (),
            stFitness.getMin ());
    fflush(NULL);
}



void DSMGA2::buildFastCounting() {
    genSelectionPool();

    for(int i = 0; i < nCurrent; ++i) {
        int winner = 0;
        double winnerFitness = -INF;
        for(int j = 0; j < selectionPressure; ++j) {
            int challenger = selectionPool[selectionPressure * i + j];
            double challengerFitness = population[challenger].getFitness();
            if(challengerFitness > winnerFitness) {
                winner = challenger;
                winnerFitness = challengerFitness;
            }
        }
        memcpy(totalGene + i * chGeneLength, population[winner].getGene(), sizeof(unsigned int) * chGeneLength);
    }

    cudaMemcpy2D(d_totalGene, tgPitch, totalGene, sizeof(unsigned int) * chGeneLength, sizeof(unsigned int) * chGeneLength, nCurrent, cudaMemcpyHostToDevice);
    deviceBuildFastCounting<<<chGeneLength, 32*fcGeneLength>>>(d_totalGene, d_fastCounting, d_selectionPool, chGeneLength, fcGeneLength, tgPitch, fcPitch, ell, nCurrent);
}

void DSMGA2::restrictedMixing(Chromosome& ch) {
    int r = myRand.uniformInt(0, ell-1);
    list<int> mask = masks[r];
    size_t size = findSize(ch, mask);
    if (size > (size_t)ell/2) {
        size = ell/2;
    }

    // prune mask to exactly size
    while (mask.size() > size) {
        mask.pop_back();
    }

    bool taken = restrictedMixing(ch, mask);

    EQ = true;
    if (taken) {
        genOrderN();
        for (int i=0; i<nCurrent; ++i) {
            if (EQ) {
                backMixingE(ch, mask, population[orderN[i]]);
            }
            else {
                backMixing(ch, mask, population[orderN[i]]);
            }
        }
    }
}

void DSMGA2::backMixing(Chromosome& source, list<int>& mask, Chromosome& des) {
    Chromosome trial(ell);
    trial = des;
    for (list<int>::iterator it = mask.begin(); it != mask.end(); ++it) {
        trial.setVal(*it, source.getVal(*it));
    }

    if (trial.getFitness() > des.getFitness()) {
        pHash.erase(des.getKey());
        pHash[trial.getKey()] = trial.getFitness();
        des = trial;
        return;
    }

}

void DSMGA2::backMixingE(Chromosome& source, list<int>& mask, Chromosome& des) {
    Chromosome trial(ell);
    trial = des;
    for (list<int>::iterator it = mask.begin(); it != mask.end(); ++it) {
        trial.setVal(*it, source.getVal(*it));
    }

    if (trial.getFitness() > des.getFitness()) {
        pHash.erase(des.getKey());
        pHash[trial.getKey()] = trial.getFitness();

        EQ = false;
        des = trial;
        return;
    }

    if (trial.getFitness() >= des.getFitness()) {
        pHash.erase(des.getKey());
        pHash[trial.getKey()] = trial.getFitness();

        des = trial;
        return;
    }

}

bool DSMGA2::restrictedMixing(Chromosome& ch, list<int>& mask) {

    bool taken = false;
    size_t lastUB = 0;

    for (size_t ub = 1; ub <= mask.size(); ++ub) {
        size_t size = 1;
        Chromosome trial(ell);
        trial = ch;

        for (list<int>::iterator it = mask.begin(); it != mask.end(); ++it) {
            trial.flip(*it);
            ++size;
            if (size > ub) {
                break;
            }
        }

        if (isInP(trial)) {
            break;
        }


        if (trial.getFitness() >= ch.getFitness()) {
            pHash.erase(ch.getKey());
            pHash[trial.getKey()] = trial.getFitness();

            taken = true;
            ch = trial;
        }

        if (taken) {
            lastUB = ub;
            break;
        }
    }

    if (lastUB != 0) {
        while (mask.size() > lastUB) {
            mask.pop_back();
        }
    }

    return taken;

}

size_t DSMGA2::findSize(Chromosome& ch, list<int>& mask) const {
    DLLA candidate(nCurrent);
    for (int i=0; i<nCurrent; ++i) {
        candidate.insert(i);
    }

    size_t size = 0;
    for (list<int>::iterator it = mask.begin(); it != mask.end(); ++it) {
        int allele = ch.getVal(*it);
        for (DLLA::iterator it2 = candidate.begin(); it2 != candidate.end(); ++it2) {
            if (population[*it2].getVal(*it) == allele) {
                candidate.erase(*it2);
            }

            if (candidate.isEmpty()) {
                break;
            }
        }

        if (candidate.isEmpty()) {
            break;
        }

        ++size;
    }

    return size;


}

size_t DSMGA2::findSize(Chromosome& ch, list<int>& mask, Chromosome& ch2) const {

    size_t size = 0;
    for (list<int>::iterator it = mask.begin(); it != mask.end(); ++it) {
        if (ch.getVal(*it) == ch2.getVal(*it)) break;
        ++size;
    }
    return size;
}

void DSMGA2::mixing() {
    cudaEventRecord(buildModelStart, 0);

    //* really learn model
    buildFastCounting();
    buildGraph();

    cudaEventRecord(buildModelStop, 0);
    cudaEventSynchronize(buildModelStop);
    cudaEventElapsedTime(&time, buildModelStart, buildModelStop);
    buildModelTime += time;

    cudaEventRecord(findCliqueStart, 0);

    for(int i = 0; i < nCurrent; ++i) {
        memcpy(totalGene + i * chGeneLength, population[i].getGene(), sizeof(unsigned int) * chGeneLength);
    }
    cudaMemcpy2D(d_totalGene, tgPitch, totalGene, sizeof(unsigned int) * chGeneLength, sizeof(unsigned int) * chGeneLength, nCurrent, cudaMemcpyHostToDevice);
    deviceFindClique<<<ell, ((ell > nCurrent) ? ell : nCurrent)>>>(d_myGraph, d_masks, d_totalGene, chGeneLength, tgPitch, ell, nCurrent, d_randStates);
    cudaMemcpy(tmp_masks, d_masks, sizeof(int) * ell * ell, cudaMemcpyDeviceToHost);

    cudaEventRecord(findCliqueStop, 0);
    cudaEventSynchronize(findCliqueStop);
    cudaEventElapsedTime(&time, findCliqueStart, findCliqueStop);
    findCliqueTime += time;

    for(int i = 0; i < ell; ++i) {
        masks[i].clear();
        for(int j = 0; j < ell; ++j) {
            if(tmp_masks[i*ell+j] == -1) {
                break;
            } else {
                masks[i].push_back(tmp_masks[i*ell+j]);
            }
        }
    }

    int repeat = (ell>50)? ell/50: 1;

    for (int k=0; k<repeat; ++k) {
        genOrderN();
        for (int i=0; i<nCurrent; ++i) {
            restrictedMixing(population[orderN[i]]);
            if (Chromosome::hit) break;
        }
        if (Chromosome::hit) break;
    }


}

inline bool DSMGA2::isInP(const Chromosome& ch) const {

    unordered_map<unsigned long, double>::const_iterator it = pHash.find(ch.getKey());
    return (it != pHash.end());
}

inline void DSMGA2::genOrderN() {
    myRand.uniformArray(orderN, nCurrent, 0, nCurrent-1);
}

inline void DSMGA2::genOrderELL() {
    myRand.uniformArray(orderELL, ell, 0, ell-1);
}

void DSMGA2::buildGraph() {
    int threadsNum = (512 > ell) ? 512 : ell;
    deviceBuildGraph<<<(graphSize/threadsNum)+1, threadsNum>>>(d_myGraph, d_fastCounting, fcPitch, fcGeneLength, ell, nCurrent, graphSize);
}

// from 1 to ell, pick by max edge
void DSMGA2::findClique(int startNode, list<int>& result) {


    result.clear();

    DLLA rest(ell);
    genOrderELL();
    for (int i=0; i<ell; ++i) {
        if (orderELL[i]==startNode)
            result.push_back(orderELL[i]);
        else
            rest.insert(orderELL[i]);
    }

    double *connection = new double[ell];

    for (DLLA::iterator iter = rest.begin(); iter != rest.end(); ++iter) {
        connection[*iter] = read(myGraph, startNode, *iter, ell);
    }

    while (!rest.isEmpty()) {
        double max = -INF;
        int index = -1;
        for (DLLA::iterator iter = rest.begin(); iter != rest.end(); ++iter) {
            if (max < connection[*iter] - EPSILON) {
                max = connection[*iter];
                index = *iter;
            }
        }

        rest.erase(index);
        result.push_back(index);

        for (DLLA::iterator iter = rest.begin(); iter != rest.end(); ++iter) {
            connection[*iter] += read(myGraph, index, *iter, ell);
        }
    }


    delete []connection;

}

double DSMGA2::computeMI(double a00, double a01, double a10, double a11) const {
    double p0 = a00 + a01;
    double q0 = a00 + a10;
    double p1 = 1-p0;
    double q1 = 1-q0;

    double join = 0.0;
    if (a00 > EPSILON) {
        join += a00*log(a00);
    }
    if (a01 > EPSILON) {
        join += a01*log(a01);
    }
    if (a10 > EPSILON) {
        join += a10*log(a10);
    }
    if (a11 > EPSILON) {
        join += a11*log(a11);
    }

    double p = 0.0;
    if (p0 > EPSILON) {
        p += p0*log(p0);
    }
    if (p1 > EPSILON) {
        p += p1*log(p1);
    }


    double q = 0.0;
    if (q0 > EPSILON) {
        q += q0*log(q0);
    }
    if (q1 > EPSILON) {
        q += q1*log(q1);
    }

    return -p-q+join;

}


void DSMGA2::selection () {
    tournamentSelection ();
}


// tournamentSelection without replacement
void DSMGA2::tournamentSelection () {
    int i, j;

    int randArray[selectionPressure * nCurrent];

    for (i = 0; i < selectionPressure; i++) {
        myRand.uniformArray (randArray + (i * nCurrent), nCurrent, 0, nCurrent - 1);
    }

    for (i = 0; i < nCurrent; i++) {
        int winner = 0;
        double winnerFitness = -INF;

        for (j = 0; j < selectionPressure; j++) {
            int challenger = randArray[selectionPressure * i + j];
            double challengerFitness = population[challenger].getFitness();

            if (challengerFitness-EPSILON > winnerFitness) {
                winner = challenger;
                winnerFitness = challengerFitness;
            }

        }
        selectionIndex[i] = winner;
    }
}

void DSMGA2::genSelectionPool() {
    for(int i = 0; i < selectionPressure; ++i) {
        myRand.uniformArray(selectionPool + (i*nCurrent), nCurrent, 0, nCurrent-1);
    }
}

__global__ void deviceBuildFastCounting(unsigned int *d_totalGene, unsigned int *d_fastCounting, int *d_selectionPool, int chGeneLength, int fcGeneLength, size_t tgPitch, size_t fcPitch, int ell, int nCurrent) {
    __shared__ unsigned int partialGenes[12288];
    unsigned int partialFastCountGene = 0;

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(threadIdx.x < nCurrent) {
        partialGenes[threadIdx.x] = (d_totalGene + threadIdx.x * tgPitch/sizeof(unsigned int))[blockIdx.x];
    }

    __syncthreads();
    int ellIdx = (idx & 31) + blockIdx.x * 32;
    int localEllIdx = idx & 31;
    int fcGeneIdx = threadIdx.x >> 5;
    if(ellIdx < ell && fcGeneIdx < fcGeneLength) {
        for(int i = 32 * fcGeneIdx; i < 32 * fcGeneIdx + 32; ++i) {
            if(i >= nCurrent) {
                break;
            }

            
            if((partialGenes[i] >> localEllIdx) & 1u == 1) {
                partialFastCountGene |= (1u << remainderInt(i));
            } else {
                partialFastCountGene &= ~(1u << remainderInt(i));
            }
            
            //setVal(partialFastCountGene + localEllIdx * fcGeneLength, i, (partialGenes[i] >> localEllIdx) & 1u);
        }
        //d_fastCounting[ellIdx * fcPitch/sizeof(unsigned int) + fcGeneIdx] = partialFastCountGene[localEllIdx * fcGeneLength + fcGeneIdx];
        d_fastCounting[ellIdx * fcPitch/sizeof(unsigned int) + fcGeneIdx] = partialFastCountGene;
    }
}

__global__ void deviceBuildGraph(double *d_myGraph, unsigned int *d_fastCounting, size_t fcPitch, int fcGeneLength, int ell, int nCurrent, int graphSize) {
    __shared__ unsigned int sharedFastCounting[12288];

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(threadIdx.x < ell) {
        for(int i = 0; i < fcGeneLength; ++i) {
            (sharedFastCounting + threadIdx.x * fcGeneLength)[i] = (d_fastCounting + threadIdx.x * fcPitch/sizeof(unsigned int))[i];
        }
    }
    __syncthreads();
    if(idx < graphSize) {
        int i = 0;
        int j = 0;
        int layerNumber = ell - 1;
        while(idx - layerNumber >= 0) {
            ++i;
            idx -= layerNumber;
            --layerNumber;
        }
        j = idx + i + 1;

        int iOnes = countOnes(sharedFastCounting + i * fcGeneLength, fcGeneLength);
        int jOnes = countOnes(sharedFastCounting + j * fcGeneLength, fcGeneLength);
        int nX = countXOR(sharedFastCounting + i * fcGeneLength, sharedFastCounting + j * fcGeneLength, fcGeneLength);
        int n11 = (iOnes + jOnes - nX) >> 1;
        int n10 = iOnes - n11;
        int n01 = jOnes - n11;
        int n00 = nCurrent - n01 - n10 - n11;

        double p00 = (double)n00/(double)nCurrent;
        double p01 = (double)n01/(double)nCurrent;
        double p10 = (double)n10/(double)nCurrent;
        double p11 = (double)n11/(double)nCurrent;

        write(d_myGraph, i, j, deviceComputeMI(p00, p01, p10, p11), ell);
    }
}

__global__ void deviceFindClique(double *d_myGraph, int *d_masks, unsigned int *d_totalGene, int chGeneLength, size_t tgPitch, int ell, int nCurrent, curandState_t *d_randStates) {
    __shared__ int buffer[12287];
    __shared__ bool hasSupply;
    int *result = &buffer[0];
    double *connections = (double *)&buffer[ell];
    int *reductionMaxIdx = &buffer[3*ell];
    int *used = (int *)&buffer[4*ell];
    unsigned int *localPopulation = (unsigned int *)&buffer[5*ell];
    int *representative = &buffer[12286];
    curandState_t localRandState;

    int idx = threadIdx.x;
    bool isDonor = true;
    result[idx] = -1;
    if(idx < ell/2) {
        localRandState = d_randStates[threadIdx.x + blockIdx.x * blockDim.x];
        if(idx == 0) {
            *representative = curand(&localRandState) % nCurrent;
        }
    }
    if(idx < nCurrent) {
        for(int i = 0; i < chGeneLength; ++i) {
            localPopulation[idx * chGeneLength + i] = d_totalGene[idx * tgPitch/sizeof(unsigned int) + i];
        }
    }
    __syncthreads();
    if(idx == blockIdx.x) {
        used[idx] = 1;
        result[0] = idx;
        connections[idx] = 0;
    } else if(idx < ell) {
        used[idx] = 0;
        connections[idx] = read(d_myGraph, blockIdx.x, idx, ell);
    }
    __syncthreads();

    for(int resultSize = 1; resultSize < ell; ++resultSize) {
        if(idx < ell) {
            reductionMaxIdx[idx] = idx;
        }
        if(idx == 0) {
            hasSupply = false;
        }
        __syncthreads();
        int reductionN = ell/2;
        while(reductionN != 0) {
            if(idx < reductionN) {
                if(connections[reductionMaxIdx[idx]] == connections[reductionMaxIdx[idx+reductionN]]) {
                    reductionMaxIdx[idx] = (curand(&localRandState) & 1 == 0) ? reductionMaxIdx[idx] : reductionMaxIdx[idx+reductionN];
                } else {
                    reductionMaxIdx[idx] = (connections[reductionMaxIdx[idx]] > connections[reductionMaxIdx[idx+reductionN]]) ? reductionMaxIdx[idx] : reductionMaxIdx[idx+reductionN];
                }
            }
            __syncthreads();
            if(idx == 0) {
                if(reductionN & 1 != 0 && reductionN != 1) {
                    if(connections[reductionMaxIdx[0]] == connections[reductionMaxIdx[reductionN-1]]) {
                        reductionMaxIdx[0] = (curand(&localRandState) & 1 == 0) ? reductionMaxIdx[0] : reductionMaxIdx[reductionN-1];
                    } else {
                        reductionMaxIdx[0] = (connections[reductionMaxIdx[0]] > connections[reductionMaxIdx[reductionN-1]]) ? reductionMaxIdx[0] : reductionMaxIdx[reductionN-1];
                    }
                }
            }
            reductionN = reductionN >> 1;
            __syncthreads();
        }

        if(idx == 0) {
            used[reductionMaxIdx[0]] = 1;
            result[resultSize] = reductionMaxIdx[0];
        }
        __syncthreads();
        if(idx < nCurrent) {
            if(isDonor) {
                if(getVal(&localPopulation[*representative * chGeneLength], reductionMaxIdx[0]) == getVal(&localPopulation[idx * chGeneLength], reductionMaxIdx[0])) {
                    isDonor = false;
                } else {
                    hasSupply = true;
                }
            }
        }
        __syncthreads();
        if(!hasSupply) {
            break;
        }
        if(idx < ell) {
            if(used[idx]) {
                connections[idx] = 0;
            } else {
                connections[idx] += read(d_myGraph, reductionMaxIdx[0], idx, ell);
            }
        }
    }
    __syncthreads();
    if(idx < ell) {
        d_masks[blockIdx.x * ell + idx] = result[idx];
        if(idx < ell/2) {
            d_randStates[threadIdx.x + blockIdx.x * blockDim.x] = localRandState;
        }
    }
}

__device__ __host__ int getVal(unsigned int *gene, int idx) {
    return ((gene[quotientInt(idx)] >> remainderInt(idx)) & 1u);
}

__device__ void setVal(unsigned int *gene, int idx, int val) {
    if(val == 1) {
        gene[quotientInt(idx)] |= (1u << remainderInt(idx));
    } else {
        gene[quotientInt(idx)] &= ~(1u << remainderInt(idx));
    }
}

__device__ int countOnes(unsigned int *gene, int fcGeneLength) {
    int n = 0;
    for(int i = 0; i < fcGeneLength; ++i) {
        n += __popc(gene[i]);
    }
    return n;
}

__device__ int countXOR(unsigned int *gene1, unsigned int *gene2, int fcGeneLength) {
    int n = 0;
    for(int i = 0; i < fcGeneLength; ++i) {
        n += __popc(gene1[i] ^ gene2[i]);
    }
    return n;
}

__device__ double deviceComputeMI(double a00, double a01, double a10, double a11) {
    double p0 = a00 + a01;
    double q0 = a00 + a10;
    double p1 = 1-p0;
    double q1 = 1-q0;

    double join = 0.0;
    if (a00 > EPSILON) {
        join += a00*__logf(a00);
    }
    if (a01 > EPSILON) {
        join += a01*__logf(a01);
    }
    if (a10 > EPSILON) {
        join += a10*__logf(a10);
    }
    if (a11 > EPSILON) {
        join += a11*__logf(a11);
    }

    double p = 0.0;
    if (p0 > EPSILON) {
        p += p0*__logf(p0);
    }
    if (p1 > EPSILON) {
        p += p1*__logf(p1);
    }

    double q = 0.0;
    if (q0 > EPSILON) {
        q += q0*__logf(q0);
    }
    if (q1 > EPSILON) {
        q += q1*__logf(q1);
    }

    double MI = -p-q+join;
    return (MI > 0) ? MI : 0;
}
