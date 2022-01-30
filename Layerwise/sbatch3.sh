#!/bin/bash
#SBATCH --mail-type=END
#SBATCH --mail-user=ma.xu1@northeastern.edu
#SBATCH -N 1
#SBATCH -p multigpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64Gb
#SBATCH --time=1-00:00:00
#SBATCH --output=%j.log


module load cuda/11.0
module unload gcc/5.5.0
module load gcc/8.1.0
export CC=/shared/centos7/gcc/8.1.0/bin/gcc
export CXX=/shared/centos7/gcc/8.1.0/bin/g++
conda activate im2vec
source activate im2vec
cd /scratch/ma.xu1/diffvg/Layerwise/


python main.py --config config/all.yaml --experiment experiment_rebuttal --signature 9 --target data/rebuttal/9.png --log_dir log/rebuttal
python main.py --config config/all.yaml --experiment experiment_rebuttal --signature 10 --target data/rebuttal/10.png --log_dir log/rebuttal
python main.py --config config/all.yaml --experiment experiment_rebuttal --signature 11 --target data/rebuttal/11.png --log_dir log/rebuttal
python main.py --config config/all.yaml --experiment experiment_rebuttal --signature 12 --target data/rebuttal/12.png --log_dir log/rebuttal
