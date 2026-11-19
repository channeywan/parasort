CXX = g++
MPICXX ?= mpicxx
NVCC ?= nvcc
CUDA_ARCH ?= sm_89
CXXFLAGS ?= -O3 -march=native -std=c++17 -fopenmp -pthread -Wall -Wextra
CPPFLAGS += -Iinclude
LDFLAGS += -fopenmp -pthread
COMMON_SRC := $(wildcard src/common/*.cpp)
COMMON_OBJ := $(COMMON_SRC:src/%.cpp=build/%.o)
HAS_MPI := $(shell command -v $(MPICXX) 2>/dev/null)
HAS_CUDA := $(shell command -v $(NVCC) 2>/dev/null)
$(info CUDA: $(if $(HAS_CUDA),enabled,disabled))
$(info MPI: $(if $(HAS_MPI),enabled,disabled))
.PHONY: all seq pthreads openmp mpi cuda test clean


all: seq pthreads openmp mpi cuda
seq: bin/seq_sort
pthreads: bin/bucket_pthreads
openmp: bin/bucket_omp bin/psrs_omp bin/ref_sorts
build/%.o: src/%.cpp $(wildcard include/*.hpp)
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@
bin/seq_sort: src/seq/seq_sort.cpp $(COMMON_OBJ)
	@mkdir -p bin
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ $(LDFLAGS) -o $@
bin/bucket_pthreads: src/pthreads/bucket_pthreads.cpp $(COMMON_OBJ)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ $(LDFLAGS) -o $@
bin/%: src/openmp/%.cpp $(COMMON_OBJ)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ $(LDFLAGS) -o $@
ifneq ($(HAS_MPI),)
mpi: bin/psrs_mpi bin/psrs_hybrid bin/pipeline_sort bin/affinity_probe
bin/affinity_probe: src/tools/affinity_probe.cpp
	@mkdir -p bin
	$(MPICXX) $(CPPFLAGS) $(CXXFLAGS) $^ $(LDFLAGS) -Wl,-rpath,'$$ORIGIN/../../.conda/lib' -o $@
bin/psrs_mpi: src/mpi/psrs_mpi.cpp src/mpi/mpi_engine.cpp $(COMMON_OBJ)
	$(MPICXX) $(CPPFLAGS) $(CXXFLAGS) $^ $(LDFLAGS) -Wl,-rpath,'$$ORIGIN/../../.conda/lib' -o $@
bin/psrs_hybrid: src/mpi/psrs_hybrid.cpp src/mpi/mpi_engine.cpp $(COMMON_OBJ)
	$(MPICXX) $(CPPFLAGS) $(CXXFLAGS) $^ $(LDFLAGS) -Wl,-rpath,'$$ORIGIN/../../.conda/lib' -o $@
bin/pipeline_sort: src/mpi/pipeline_sort.cpp $(COMMON_OBJ)
	$(MPICXX) $(CPPFLAGS) $(CXXFLAGS) $^ $(LDFLAGS) -Wl,-rpath,'$$ORIGIN/../../.conda/lib' -o $@
else
mpi:
	@echo "SKIP MPI: mpicxx not found (use environment.yml)"
endif
ifneq ($(HAS_CUDA),)
cuda: bin/gpu_sort bin/gpu_multi
bin/%: src/cuda/%.cu $(COMMON_OBJ) $(wildcard include/*.cuh)
	$(NVCC) -O3 -std=c++17 -arch=$(CUDA_ARCH) -ccbin /usr/bin/g++ -Xcompiler -fopenmp -Xcompiler -pthread -Iinclude $< $(COMMON_OBJ) -o $@
else
cuda:
	@echo "SKIP CUDA: nvcc not found"
endif
test: all
	bash tests/test_correctness.sh --quick
clean:
	rm -rf build bin
	mkdir -p bin

bin/test_core: tests/test_core.cpp $(COMMON_OBJ)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ $(LDFLAGS) -o $@
