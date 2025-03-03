# HRPF

## Run

```shell
mkdir build & cd build
cmake ..
make
```

## Test

install python and dependencies and run:

```shell
python test.py
```

the output is an excel file.

## Todos

### Missing Codes

- Merge Sort: StarPU, OpenMP, GPU
- Strassen-Winograd: StarPU, OpenMP, MKL, CUBLAS
- Hadamard: HRPF

### Test

1. knn, kmeans are not better than the original version.
2. AdjointConv takes too long time to run.
3. Strassen-Winograd is not implemented.
4. Hadamard is not implemented.
5. Merge Sort is not implemented.

### AutoTest Script

