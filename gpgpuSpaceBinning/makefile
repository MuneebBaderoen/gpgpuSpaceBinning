CUDA_INSTALL_PATH ?= /usr/local/cuda

CXX := g++
CC := gcc
LINK := g++ -fPIC
NVCC  := nvcc -arch sm_20 -ccbin /usr/bin

# Includes
INCLUDES = -I. -I$(CUDA_INSTALL_PATH)/include

# Common flags
COMMONFLAGS += $(INCLUDES)
NVCCFLAGS += $(COMMONFLAGS)
CXXFLAGS += $(COMMONFLAGS)
CFLAGS += $(COMMONFLAGS)

LIB_CUDA := -L$(CUDA_INSTALL_PATH)/lib64 -lcudart -lrt

# Project Specific things
OBJS = main.cu.o
TARGET = spaceBinning 
LINKLINE = $(LINK) -o $(TARGET) $(OBJS) $(LIB_CUDA)


.SUFFIXES: .c .cpp .cu .o

%.c.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.cu.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

%.cpp.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TARGET): $(OBJS)
	$(LINKLINE)


run: $(TARGET)
	qsub qsubScript; watch qstat

clean:
	rm *.o $(TARGET) munGpu.*
