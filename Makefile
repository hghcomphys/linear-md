# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -O2 -Wall -Wextra -std=c++11

# Target executable 
TARGET = md.x

# Source files
SRC = md.cpp

# Build rule
all: $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

# Clean rule
clean:
	rm -f $(TARGET) 

