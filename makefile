# We will benchmark you against Intel MKL implementation, the default processor vendor-tuned implementation.
# This makefile is intended for the Intel C compiler.
# Your code must compile (with icc) with the given CFLAGS. You may experiment with the OPT variable to invoke additional compiler options.

CC = clang++
# OPT = -no-multibyte-chars
FLAGS = -O3
DEBUG = -g --debug
# -fopt-info
PREFIX= /home/pzzzzz/
LDLIBS = -I$(PREFIX)miniconda3/envs/new/include -I$(PREFIX)Openblas-install/include -I$(PREFIX)MyProjects/CPU-pytorch/CPU_pytorch/ops/csrc/include -L$(PREFIX)miniconda3/envs/new/lib -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl -std=c++17 -march=native -Xpreprocessor -fopenmp
targets = benchmark-naive benchmark-blas
objects = benchmark.o gemm-naive.o test.o


.PHONY : default
default : all

.PHONY : all
all : clean $(targets)

benchmark-naive : benchmark.o gemm-naive-global-all.o
	$(CC) -o $@ $^ $(FLAGS)  $(LDLIBS) -rpath $(PREFIX)miniconda3/envs/new/lib
benchmark-all : benchmark-all.o gemm-naive-global-all.o
	$(CC) -o $@ $^ $(FLAGS)  $(LDLIBS) -rpath $(PREFIX)miniconda3/envs/new/lib
benchmark-all-blas : benchmark-all.o gemm-blas.o
	$(CC) -o $@ $^ $(FLAGS)  $(LDLIBS) -rpath $(PREFIX)miniconda3/envs/new/lib
benchmark-naive-d : benchmark-d.o gemm-naive-global-d.o
	$(CC) -o $@ $^ -g --debug  $(LDLIBS) -rpath $(PREFIX)miniconda3/envs/new/lib
benchmark-no-packing: benchmark-all-no-packing.o gemm-naive-global-all.o
	$(CC) -o $@ $^ $(FLAGS)
benchmark-no-packing-d: benchmark-all-no-packing-d.o gemm-naive-global-all-d.o
	$(CC) -o $@ $^ $(DEBUG)
benchmark-kn: benchmark-all-no-packing.o gemm-naive-global-all-kn.o
	$(CC) -o $@ $^ $(FLAGS)
benchmark-kn-d: benchmark-all-no-packing-d.o gemm-naive-global-all-kn-d.o
	$(CC) -o $@ $^ $(DEBUG)
benchmark-kn-omp: benchmark-all-no-packing.o gemm-naive-global-all-kn.o
	$(CC) -o $@ $^ $(FLAGS) -lomp
benchmark-no-packing-omp: benchmark-all-no-packing.o gemm-naive-global-all.o
	$(CC) -o $@ $^ $(FLAGS) -lomp
benchmark-no-packing-blas: benchmark-all-no-packing.o gemm-blas.o
	$(CC) -o $@ $^ $(FLAGS)  $(LDLIBS) -rpath $(PREFIX)miniconda3/envs/new/lib
benchmark-no-packing-openblas: benchmark-all.o gemm-openblas.o
	$(CC) -o $@ $^ $(FLAGS)  $(LDLIBS) -rpath $(PREFIX)Openblas-install/lib -rpath $(PREFIX)miniconda3/envs/new/lib
benchmark-all-naive : benchmark-all.o gemm-naive-global-all.o
	$(CC) -o $@ $^ $(FLAGS)  $(LDLIBS) -rpath $(PREFIX)miniconda3/envs/new/lib
benchmark-omp : benchmark.o gemm-naive.o
	$(CC) -o $@ $^ $(FLAGS) -D__OMP__  $(LDLIBS) -rpath $(PREFIX)miniconda3/envs/new/lib
benchmark-blas : benchmark.o gemm-blas.o
	$(CC) -o $@ $^ $(FLAGS)  $(LDLIBS) -rpath $(PREFIX)miniconda3/envs/new/lib
benchmark-openblas : benchmark.o gemm-openblas.o
	$(CC) -o $@ $^ $(FLAGS)  $(LDLIBS) -rpath $(PREFIX)OpenBLAS-0.3.20-x64/lib
benchmark-wsm : benchmark.o gemm-wsm.o
	$(CC) -o $@ $^ $(FLAGS)  $(LDLIBS) -rpath $(PREFIX)miniconda3/envs/new/lib
benchmark-blis : benchmark.o gemm-blis.o
	$(CC) -o $@ $^ $(FLAGS)  $(LDLIBS) $(PREFIX)blis-install/lib/libblis.a -rpath $(PREFIX)blis-install/lib -rpath $(PREFIX)miniconda3/envs/new/lib
benchmark-ulm : benchmark.o gemm-ulm.o
	$(CC) -o $@ $^ $(FLAGS)  $(LDLIBS) -rpath $(PREFIX)miniconda3/envs/new/lib
test : test-all-d.o gemm-naive-global-all-d.o
	$(CC) -o $@ $^ --debug -g $(LDLIBS) -rpath $(PREFIX)miniconda3/envs/new/lib
test-no-packing : test-no-packing-d.o gemm-naive-global-all-d.o
	$(CC) -o $@ $^ --debug -g $(LDLIBS) -rpath $(PREFIX)miniconda3/envs/new/lib
test-m : mkl_test.o gemm-blas.o
	$(CC) -o $@ $^ --debug -g $(LDLIBS) -rpath $(PREFIX)miniconda3/envs/new/lib
%.o : %.cpp
	$(CC) -c $(CFLAGS) $(FLAGS) $< $(LDLIBS)

%-d.o:%.cpp
	$(CC) -c $(CFLAGS) $(DEBUG) $< $(LDLIBS) -o $@

run-naive:
	make benchmark-naive
	export OMP_NUM_THREAnew=1
	mv output.log old.log
	./benchmark-naive > output.log
	rm benchmark-naive
	python plot.py

run-blas:
	make benchmark-blas
	export OMP_NUM_THREAnew=1
	./benchmark-blas > mkl.log
	rm benchmark-blas
	python plot.py

.PHONY : clean
clean:
	rm -f $(targets) $(objects)