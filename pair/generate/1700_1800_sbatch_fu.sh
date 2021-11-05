#!/bin/bash
#SBATCH --mail-type=END
#SBATCH --mail-user=ma.xu1@northeastern.edu
#SBATCH -N 1
#SBATCH -p fugpu
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --output=%j.log

module load cuda/11.0
module unload gcc/5.5.0
module load gcc/8.1.0
export CC=/shared/centos7/gcc/8.1.0/bin/gcc
export CXX=/shared/centos7/gcc/8.1.0/bin/g++
source activate im2vec

cd /scratch/ma.xu1/diffvg/pair/generate
python generate_pair.py  --pool_size 60  --free --start 1700 --end 1800
