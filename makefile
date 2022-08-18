# We will benchmark you against Intel MKL implementation, the default processor vendor-tuned implementation.
# This makefile is intended for the Intel C compiler.
# Your code must compile (with icc) with the given CFLAGS. You may experiment with the OPT variable to invoke additional compiler options.

CC = g++
# OPT = -no-multibyte-chars
FLAGS = -O3
DEBUG = -g --debug
# -fopt-info
LDLIBS = -I/Users/pzzzzz/miniconda3/envs/ds/include -I/Users/pzzzzz/blis-install/include -I/Users/pzzzzz/CPU-pytorch/CPU_pytorch/ops/csrc/include -L/Users/pzzzzz/miniconda3/envs/ds/lib -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl -std=c++17 -march=native -Xpreprocessor -fopenmp
targets = benchmark-naive benchmark-blas
objects = benchmark.o gemm-naive.o test.o


.PHONY : default
default : all

.PHONY : all
all : clean $(targets)

benchmark-naive : benchmark.o gemm-naive-global.o
	$(CC) -o $@ $^ $(FLAGS)  $(LDLIBS) -rpath /Users/pzzzzz/miniconda3/envs/ds/lib
benchmark-naive-d : benchmark.o gemm-naive.o
	$(CC) -o $@ $^ -g --debug  $(LDLIBS) -rpath /Users/pzzzzz/miniconda3/envs/ds/lib
benchmark-no-packing: benchmark-all-no-packing.o gemm-naive-global-all.o
	$(CC) -o $@ $^ $(FLAGS)  $(LDLIBS) -rpath /Users/pzzzzz/miniconda3/envs/ds/lib
benchmark-no-packing-blas: benchmark-all-no-packing.o gemm-blas.o
	$(CC) -o $@ $^ $(FLAGS)  $(LDLIBS) -rpath /Users/pzzzzz/miniconda3/envs/ds/lib
benchmark-all-naive : benchmark-all.o gemm-naive-global-all.o
	$(CC) -o $@ $^ $(FLAGS)  $(LDLIBS) -rpath /Users/pzzzzz/miniconda3/envs/ds/lib
benchmark-omp : benchmark.o gemm-naive.o
	$(CC) -o $@ $^ $(FLAGS) -D__OMP__  $(LDLIBS) -rpath /Users/pzzzzz/miniconda3/envs/ds/lib
benchmark-blas : benchmark.o gemm-blas.o
	$(CC) -o $@ $^ $(FLAGS)  $(LDLIBS) -rpath /Users/pzzzzz/miniconda3/envs/ds/lib
benchmark-openblas : benchmark.o gemm-openblas.o
	$(CC) -o $@ $^ $(FLAGS)  $(LDLIBS) -rpath /Users/pzzzzz/OpenBLAS-0.3.20-x64/lib
benchmark-wsm : benchmark.o gemm-wsm.o
	$(CC) -o $@ $^ $(FLAGS)  $(LDLIBS) -rpath /Users/pzzzzz/miniconda3/envs/ds/lib
benchmark-blis : benchmark.o gemm-blis.o
	$(CC) -o $@ $^ $(FLAGS)  $(LDLIBS) /Users/pzzzzz/blis-install/lib/libblis.a -rpath /Users/pzzzzz/blis-install/lib -rpath /Users/pzzzzz/miniconda3/envs/ds/lib
benchmark-ulm : benchmark.o gemm-ulm.o
	$(CC) -o $@ $^ $(FLAGS)  $(LDLIBS) -rpath /Users/pzzzzz/miniconda3/envs/ds/lib
test : test-all.o gemm-naive-global-all.o
	$(CC) -o $@ $^ --debug -g $(LDLIBS) /Users/pzzzzz/blis-install/lib/libblis.a -rpath /Users/pzzzzz/miniconda3/envs/ds/lib
test-m : mkl_test.o gemm-blas.o
	$(CC) -o $@ $^ --debug -g $(LDLIBS) -rpath /Users/pzzzzz/miniconda3/envs/ds/lib
%.o : %.cpp
	$(CC) -c $(CFLAGS) $(FLAGS) $< $(LDLIBS)

run-naive:
	make benchmark-naive
	export OMP_NUM_THREADS=1
	mv output.log old.log
	./benchmark-naive > output.log
	rm benchmark-naive
	python plot.py

run-blas:
	make benchmark-blas
	export OMP_NUM_THREADS=1
	./benchmark-blas > mkl.log
	rm benchmark-blas
	python plot.py

.PHONY : clean
clean:
	rm -f $(targets) $(objects)