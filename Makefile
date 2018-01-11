CXX:=CC
#CXX:=pgc++
OPTFLAGS:=-qopenmp -O3 -dynamic -std=c++11 -xMIC-AVX512 -fp-model fast=2
#OPTFLAGS:=-fopenmp -O3
#OPTFLAG:=-O3
DEBUGFLAGS:=-g #-dynamic
CPPFLAGS:=$(OPTFLAGS) $(DEBUGFLAGS) -mkl
LDFLAGS:=$(DEBUGFLAGS) $(CRAY_PMI_POST_LINK_OPTS) #-L/project/projectdirs/mpccc/tkurth/NESAP/intelcaffe_internal/intelcaffe_src/external/mkl/mklml_lnx_2017.0.2.20170110/lib -lmklml_intel
LIBS:= -lpmi # $(DDT_LINK_DMALLOC)

#targets
%.o: %.cpp
	$(CXX) $(CPPFLAGS) -c $< -o $@

collectives.x : utils.o collectives.o
	$(CXX) $(CPPFLAGS) $(LDFLAGS) utils.o collectives.o -o collectives.x $(LIBS)

memmap.x : utils.o memmap.o
	$(CXX) $(CPPFLAGS) $(LDFLAGS) utils.o memmap.o -o memmap.x $(LIBS) -lrt

minibench.x : utils.o minibench.o
	$(CXX) $(CPPFLAGS) $(LDFLAGS) utils.o minibench.o -o minibench.x $(LIBS)

nn_exchange.x : utils.o nn_exchange.o
	$(CXX) $(CPPFLAGS) $(LDFLAGS) utils.o nn_exchange.o -o nn_exchange.x $(LIBS)

stream.x : stream.o
	$(CXX) $(CPPFLAGS) $(LDFLAGS) stream.o -o stream.x $(LIBS)

all: collectives.x minibench.x nn_exchange.x stream.x

.PHONY: clean
clean :
	rm -f *.o *.x
