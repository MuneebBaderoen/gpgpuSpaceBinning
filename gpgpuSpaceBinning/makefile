CC = g++

SpaceBinning: main.o
	${CC} -o SpaceBinning main.o

main.o: main.cpp
	${CC} main.cpp -c
	
clean:
	rm -r *.o SpaceBinning

run:
	./SpaceBinning
