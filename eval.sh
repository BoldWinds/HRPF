#!/usr/bin/zsh
MAX_RUN=10

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


