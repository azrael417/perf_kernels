CXX:=CC
CPPFLAGS:=-qopenmp

%.o: %.cpp
	$(CXX) $(CPPFLAGS) -c $< -o $@

nn_exchange.x : input.o nn_exchange.o
	$(CXX) $(CPPFLAGS) input.o nn_exchange.o -o nn_exchange.x

clean :
	rm -f *.o *.x
