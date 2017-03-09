CXX:=CC
OPTFLAGS:=-qopenmp -O3
DEBUGFLAGS:=-g -dynamic
CPPFLAGS:=$(OPTFLAGS) $(DEBUGFLAGS)
LDFLAGS:=$(DEBUGFLAGS)
LIBS:= # $(DDT_LINK_DMALLOC)

%.o: %.cpp
	$(CXX) $(CPPFLAGS) -c $< -o $@

nn_exchange.x : input.o nn_exchange.o
	$(CXX) $(CPPFLAGS) $(LDFLAGS) input.o nn_exchange.o -o nn_exchange.x $(LIBS)

clean :
	rm -f *.o *.x
