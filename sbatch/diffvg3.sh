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
cd /scratch/ma.xu1/diffvg/pair/Layerwise/



python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/1.png --num_paths 8 --save_folder rebuttal/1_path8
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/2.png --num_paths 8 --save_folder rebuttal/2_path8
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/3.png --num_paths 8 --save_folder rebuttal/3_path8
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/4.png --num_paths 8 --save_folder rebuttal/4_path8
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/5.png --num_paths 8 --save_folder rebuttal/5_path8
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/6.png --num_paths 8 --save_folder rebuttal/6_path8
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/7.png --num_paths 8 --save_folder rebuttal/7_path8
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/8.png --num_paths 8 --save_folder rebuttal/8_path8
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/9.png --num_paths 8 --save_folder rebuttal/9_path8
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/10.png --num_paths 8 --save_folder rebuttal/10_path8
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/11.png --num_paths 8 --save_folder rebuttal/11_path8
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/12.png --num_paths 8 --save_folder rebuttal/12_path8
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/13.png --num_paths 8 --save_folder rebuttal/13_path8
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/14.png --num_paths 8 --save_folder rebuttal/14_path8
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/15.png --num_paths 8 --save_folder rebuttal/15_path8


python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/1.png --num_paths 16 --save_folder rebuttal/1_path16
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/2.png --num_paths 16 --save_folder rebuttal/2_path16
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/3.png --num_paths 16 --save_folder rebuttal/3_path16
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/4.png --num_paths 16 --save_folder rebuttal/4_path16
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/5.png --num_paths 16 --save_folder rebuttal/5_path16
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/6.png --num_paths 16 --save_folder rebuttal/6_path16
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/7.png --num_paths 16 --save_folder rebuttal/7_path16
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/8.png --num_paths 16 --save_folder rebuttal/8_path16
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/9.png --num_paths 16 --save_folder rebuttal/9_path16
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/10.png --num_paths 16 --save_folder rebuttal/10_path16
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/11.png --num_paths 16 --save_folder rebuttal/11_path16
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/12.png --num_paths 16 --save_folder rebuttal/12_path16
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/13.png --num_paths 16 --save_folder rebuttal/13_path16
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/14.png --num_paths 16 --save_folder rebuttal/14_path16
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/15.png --num_paths 16 --save_folder rebuttal/15_path16

