# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -O3 -march=native -Wall -Wextra -std=c++11

# Target executable 
TARGET = md.x

# Source files
SRC = md.cpp

SRC: $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

run: SRC $(TARGET)
	time ./$(TARGET)

profile: $(SRC) 
	$(CXX) $(CXXFLAGS) -pg  -o $(TARGET) $(SRC)
	perf stat -e cycles,instructions,cache-references,cache-misses ./$(TARGET)

profile2: $(TARGET)
	sudo perf record ./$(TARGET)
	sudo perf report 

.PHONY: all
all: SRC

.PHONY: clean 
clean:
	rm -f $(TARGET) 
	rm -f perf.data*


