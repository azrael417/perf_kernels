CXX:=CC
OPTFLAGS:=-qopenmp -O3
DEBUGFLAGS:=-g #-dynamic
CPPFLAGS:=$(OPTFLAGS) $(DEBUGFLAGS) -mkl -std=c++11 -xMIC-AVX512 -fp-model fast=2
LDFLAGS:=$(DEBUGFLAGS) -Wl,--whole-archive,-ldmapp,--no-whole-archive
LIBS:= # $(DDT_LINK_DMALLOC)

#targets
%.o: %.cpp
	$(CXX) $(CPPFLAGS) -c $< -o $@

collectives.x : utils.o collectives.o
	$(CXX) $(CPPFLAGS) $(LDFLAGS) utils.o collectives.o -o collectives.x $(LIBS)

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
