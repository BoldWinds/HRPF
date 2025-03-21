#!/usr/bin/zsh
MAX_RUN=$1

# echo "MergeSort"
# echo "LENGTH,SEQ,STD,GPU,THRUST,HRPF"
# for (( LENGTH=10485670; LENGTH<=104856700; LENGTH+=10485670 )); do
#     echo $LENGTH
#     ./build/bin/merge_sort/ms_seq $LENGTH $MAX_RUN
#     ./build/bin/merge_sort/ms_std $LENGTH $MAX_RUN
#     ./build/bin/merge_sort/ms_gpu $LENGTH $MAX_RUN
#     ./build/bin/merge_sort/ms_thrust $LENGTH $MAX_RUN
#     ./build/bin/merge_sort/ms_hrpf $LENGTH "BBBBBBBBBB"
# done

# echo "strassen"
# echo "LENGTH,SEQ,STD,GPU,THRUST,HRPF"
# for (( LENGTH=10485670; LENGTH<=104856700; LENGTH+=10485670 )); do
#     echo $LENGTH
#     ./build/bin/strassen/matmul_omp     $LENGTH $MAX_RUN
#     ./build/bin/strassen/matmul_cublas  $LENGTH $MAX_RUN
#     ./build/bin/strassen/matmul_starpu  $LENGTH $MAX_RUN
#     ./build/bin/strassen/matmul_mkl     $LENGTH $MAX_RUN
#     ./build/bin/strassen/strassen_hrpf  $LENGTH "BBBBBBBBBB"
# done


echo "hadamard"
echo "LENGTH,SEQ,OMP,HRPF"
for (( LENGTH=416; LENGTH<=2720; LENGTH+=256)); do
    echo $LENGTH
    ./build/bin/hadamard/hadamard_seq $LENGTH
    ./build/bin/hadamard/hadamard_omp $LENGTH
    ./build/bin/hadamard/hadamard_hrpf $LENGTH
done
echo ""

echo "knn"
echo "LENGTH,SEQ,OMP,HRPF"
for (( LENGTH=35840; LENGTH<=115712; LENGTH+=8192 )); do
    echo $LENGTH
    ./build/bin/knn/knn_seq $LENGTH
    ./build/bin/knn/knn_omp $LENGTH
    ./build/bin/knn/knn_hrpf $LENGTH
done
echo ""

echo "dft"
echo "LENGTH,SEQ,OMP,HRPF"
for (( LENGTH=1024; LENGTH<=29696; LENGTH+=4096 )); do
    echo $LENGTH
    ./build/bin/dft/dft_seq $LENGTH $MAX_RUN
    ./build/bin/dft/dft_omp $LENGTH $MAX_RUN
    ./build/bin/dft/dft_hrpf $LENGTH $MAX_RUN
done
echo ""

echo "mvm"
echo "LENGTH,SEQ,OMP,HRPF"
for (( LENGTH=416; LENGTH<=2720; LENGTH+=256 )); do
    echo $LENGTH
    ./build/bin/mat_vec_mul/mvm_seq $LENGTH $MAX_RUN
    ./build/bin/mat_vec_mul/mvm_omp $LENGTH $MAX_RUN
    ./build/bin/mat_vec_mul/mvm_hrpf $LENGTH $MAX_RUN
done
echo ""

echo "Nbody"
echo "LENGTH,SEQ,OMP,HRPF"
for (( LENGTH=1024; LENGTH<=19456; LENGTH+=2048 )); do
    echo $LENGTH
    ./build/bin/nbody/nbody_seq $LENGTH $MAX_RUN
    ./build/bin/nbody/nbody_omp $LENGTH $MAX_RUN
    ./build/bin/nbody/nbody_hrpf $LENGTH $MAX_RUN
done
echo ""

echo "TRANSPOSE"
echo "LENGTH,SEQ,OMP,HRPF"
for (( LENGTH=1024; LENGTH<=8192; LENGTH+=1024 )); do
    echo $LENGTH
    ./build/bin/transpose/transpose_seq $LENGTH $MAX_RUN
    ./build/bin/transpose/transpose_omp $LENGTH $MAX_RUN
    ./build/bin/transpose/transpose_hrpf $LENGTH $MAX_RUN
done
echo ""

echo "AdjointConv"
echo "LENGTH,SEQ,OMP,HRPF"
for (( LENGTH=1024; LENGTH<=57600; LENGTH+=4096 )); do
    echo $LENGTH
    ./build/bin/adjoint_conv/ac_seq $LENGTH
    ./build/bin/adjoint_conv/ac_omp $LENGTH
    ./build/bin/adjoint_conv/ac_hrpf $LENGTH
done
echo ""

echo "kmeans"
echo "LENGTH,SEQ,OMP,HRPF"
for (( LENGTH=1024; LENGTH<=8192; LENGTH+=1024 )); do
    echo $LENGTH
    ./build/bin/kmeans/kmeans_seq $LENGTH
    ./build/bin/kmeans/kmeans_omp $LENGTH
    ./build/bin/kmeans/kmeans_hrpf $LENGTH
done