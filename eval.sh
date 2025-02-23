#!/usr/bin/zsh
MAX_RUN=$1

echo "dft"
echo "LENGTH,SEQ,OMP,HRPF"
for (( LENGTH=1024; LENGTH<=19456; LENGTH+=4096 )); do
    echo $LENGTH
    ./build/bin/dft/dft_seq $LENGTH $MAX_RUN
    ./build/bin/dft/dft_omp $LENGTH $MAX_RUN
    ./build/bin/dft/dft_hrpf $LENGTH $MAX_RUN
done

echo "mvm"
echo "LENGTH,SEQ,OMP,HRPF"
for (( LENGTH=416; LENGTH<=2720; LENGTH+=256 )); do
    echo $LENGTH
    ./build/bin/mat_vec_mul/mvm_seq $LENGTH $MAX_RUN
    ./build/bin/mat_vec_mul/mvm_omp $LENGTH $MAX_RUN
    ./build/bin/mat_vec_mul/mvm_hrpf $LENGTH $MAX_RUN
done

echo "Nbody"
echo "LENGTH,SEQ,OMP,HRPF"
for (( LENGTH=1024; LENGTH<=19456; LENGTH+=2048 )); do
    echo $LENGTH
    ./build/bin/nbody/nbody_seq $LENGTH $MAX_RUN
    ./build/bin/nbody/nbody_omp $LENGTH $MAX_RUN
    ./build/bin/nbody/nbody_hrpf $LENGTH $MAX_RUN
done

echo "TRANSPOSE"
echo "LENGTH,SEQ,OMP,HRPF"
for (( LENGTH=1024; LENGTH<=8192; LENGTH+=1024 )); do
    echo $LENGTH
    ./build/bin/transpose/transpose_seq $LENGTH $MAX_RUN
    ./build/bin/transpose/transpose_omp $LENGTH $MAX_RUN
    ./build/bin/transpose/transpose_hrpf $LENGTH $MAX_RUN
done


echo " AdjointConv"
echo "LENGTH,SEQ,OMP,HRPF"
for (( LENGTH=1024; LENGTH<=57600; LENGTH+=4096 )); do
    echo $LENGTH
    ./build/bin/adjoint_conv/ac_seq $LENGTH $MAX_RUN
    ./build/bin/adjoint_conv/ac_omp $LENGTH $MAX_RUN
    ./build/bin/adjoint_conv/ac_hrpf $LENGTH $MAX_RUN
done