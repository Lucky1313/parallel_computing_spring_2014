
BIN := main

CC=gcc
CXX=g++
RM=rm -f

CUDA_INSTALL_PATH=/usr/local/cuda-6.0
CUDA=$(CUDA_INSTALL_PATH)/bin/nvcc
CUDA_LIBS=-L"$(CUDA_INSTALL)/lib64"

CXXFLAGS=-g -O3 -Wall
CUDAFLAGS=-g -G -O3 -arch=sm_20 --ptxas-options=-v

PROJECT_PATH=.

INCLUDE= -I"$(PROJECT_PATH)/include" -I"$(CUDA_INSTALL_PATH)/include"
SRC_DIR=$(PROJECT_PATH)/src
BUILD_DIR=$(PROJECT_PATH)/build

CPP_SRCS=$(wildcard $(SRC_DIR)/*.cpp)
CU_SRCS=$(wildcard $(SRC_DIR)/*.cu)

CPP_OBJS=$(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(CPP_SRCS))
CU_OBJS=$(patsubst $(SRC_DIR)/%.cu, $(BUILD_DIR)/%.o, $(CU_SRCS))

all: $(BUILD_DIR)/$(BIN)

$(BUILD_DIR)/%.o : $(SRC_DIR)/%.cu
	$(CUDA) $(CUDAFLAGS) $(INCLUDE) -o $@ -c $<

$(BUILD_DIR)/%.o : $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE) -o $@ -c $<

$(BUILD_DIR)/$(BIN) : $(CPP_OBJS) $(CU_OBJS)
	$(CUDA) -o $(BUILD_DIR)/$(BIN) $(CU_OBJS) $(CPP_OBJS) $(INCLUDE) $(CUDA_LIBS)

clean:
	$(RM) $(BUILD_DIR)/$(BIN) $(BUILD_DIR)/*.o $(BUILD_DIR)/*.cu_o
