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


python main.py --config config/all.yaml --experiment experiment_rebuttal --signature 13 --target data/rebuttal/13.png --log_dir log/rebuttal
python main.py --config config/all.yaml --experiment experiment_rebuttal --signature 14 --target data/rebuttal/14.png --log_dir log/rebuttal
python main.py --config config/all.yaml --experiment experiment_rebuttal --signature 15 --target data/rebuttal/15.png --log_dir log/rebuttal
