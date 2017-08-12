CXX = nvcc
CXXFLAGS = -O2 -std=c++11 -Wno-deprecated-gpu-targets
#CXXFLAGS = -g -G -std=c++11 -Wno-deprecated-gpu-targets
INCLUDE = 
TLIB = -lm

#-----Suffix Rules---------------------------
# set up C++ suffixes and relationship between .cu and .o files

.SUFFIXES: .o .cu

.cu.o:
	$(CXX) $(CXXFLAGS) $(INCLUDE) -dc $<

#-----File Dependencies----------------------

SRC = $(SRC1) $(SRC2) $(SRC3)

SRC1 = chromosome.cpp dsmga2.cu fastcounting.cpp global.cu main.cu mt19937ar.cpp myrand.cpp spin.cpp nk-wa.cpp sat.cpp
SRC2 = chromosome.cpp dsmga2.cu fastcounting.cpp global.cu sweep.cu mt19937ar.cpp myrand.cpp spin.cpp nk-wa.cpp sat.cpp
SRC3 = genZobrist.cpp

OBJ = $(addsuffix .o, $(basename $(SRC)))

OBJ1 = $(addsuffix .o, $(basename $(SRC1)))
OBJ2 = $(addsuffix .o, $(basename $(SRC2)))
OBJ3 = $(addsuffix .o, $(basename $(SRC3)))

all: DSMGA2 sweep genZobrist

DSMGA2: $(OBJ1)
	$(CXX) $(CXXFLAGS) $(INCLUDE) $(TLIB) -o $@ $(OBJ1)
sweep: $(OBJ2) DSMGA2
	$(CXX) $(CXXFLAGS) $(INCLUDE) $(TLIB) -o $@ $(OBJ2)
genZobrist: $(OBJ3)
	$(CXX) $(CXXFLAGS) $(INCLUDE) $(TLIB) -o $@ $(OBJ3)

#-----Other stuff----------------------------

depend:
	makedepend -Y. $(SRC)

.PHONY: clean
clean:
	rm -f $(OBJ)

chromosome.o: spin.h chromosome.h global.cuh myrand.h mt19937ar.h bitwisedistance.h nk-wa.h doublelinkedlistarray.h zkey.h sat.h
dsmga2.o: chromosome.h global.cuh myrand.h mt19937ar.h bitwisedistance.h spin.h nk-wa.h doublelinkedlistarray.h zkey.h sat.h dsmga2.cuh statistics.h mygraph.cuh
fastcounting.o: global.cuh myrand.h mt19937ar.h bitwisedistance.h spin.h nk-wa.h doublelinkedlistarray.h zkey.h sat.h fastcounting.h
global.o: myrand.h mt19937ar.h statistics.h doublelinkedlistarray.h zkey.h chromosome.h global.cuh bitwisedistance.h spin.h nk-wa.h sat.h
main.o: statistics.h dsmga2.cuh chromosome.h global.cuh myrand.h mt19937ar.h bitwisedistance.h spin.h nk-wa.h doublelinkedlistarray.h zkey.h sat.h trimatrix.h fastcounting.h
myrand.o: myrand.h mt19937ar.h
spin.o: global.cuh myrand.h mt19937ar.h bitwisedistance.h spin.h nk-wa.h doublelinkedlistarray.h zkey.h sat.h
nk-wa.o: nk-wa.h
sat.o: sat.h
