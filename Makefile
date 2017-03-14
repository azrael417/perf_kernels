CXX:=CC
OPTFLAGS:=-qopenmp -O3
DEBUGFLAGS:=-g -dynamic
CPPFLAGS:=$(OPTFLAGS) $(DEBUGFLAGS)
LDFLAGS:=$(DEBUGFLAGS)
LIBS:= # $(DDT_LINK_DMALLOC)

%.o: %.cpp
	$(CXX) $(CPPFLAGS) -c $< -o $@

all-to-all.x : utils.o all-to-all.o
	$(CXX) $(CPPFLAGS) $(LDFLAGS) utils.o all-to-all.o -o all-to-all.x $(LIBS)

nn_exchange.x : utils.o nn_exchange.o
	$(CXX) $(CPPFLAGS) $(LDFLAGS) utils.o nn_exchange.o -o nn_exchange.x $(LIBS)

clean :
	rm -f *.o *.x
