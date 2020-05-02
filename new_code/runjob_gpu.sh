#!/usr/bin/env bash
#SBATCH -p wacc
#SBATCH -t 0-00:01:00
#SBATCH -J BlankSchmitz
#SBATCH --gres=gpu:1 -c 1

module load cuda

nvcc wrapper.cu readLVM.cpp writeCSV.cpp writeCSVInt.cpp butterHP.cpp butterLP.cpp filtfilt.cpp sort.cpp xcorr.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -o wrapper

./wrapper

echo "Job complete"