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



python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/1.png --num_paths 32 --save_folder rebuttal/1_path32
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/2.png --num_paths 32 --save_folder rebuttal/2_path32
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/3.png --num_paths 32 --save_folder rebuttal/3_path32
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/4.png --num_paths 32 --save_folder rebuttal/4_path32
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/5.png --num_paths 32 --save_folder rebuttal/5_path32
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/6.png --num_paths 32 --save_folder rebuttal/6_path32
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/7.png --num_paths 32 --save_folder rebuttal/7_path32
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/8.png --num_paths 32 --save_folder rebuttal/8_path32
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/9.png --num_paths 32 --save_folder rebuttal/9_path32
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/10.png --num_paths 32 --save_folder rebuttal/10_path32
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/11.png --num_paths 32 --save_folder rebuttal/11_path32
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/12.png --num_paths 32 --save_folder rebuttal/12_path32
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/13.png --num_paths 32 --save_folder rebuttal/13_path32
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/14.png --num_paths 32 --save_folder rebuttal/14_path32
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/15.png --num_paths 32 --save_folder rebuttal/15_path32



python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/1.png --num_paths 64 --save_folder rebuttal/1_path64
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/2.png --num_paths 64 --save_folder rebuttal/2_path64
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/3.png --num_paths 64 --save_folder rebuttal/3_path64
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/4.png --num_paths 64 --save_folder rebuttal/4_path64
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/5.png --num_paths 64 --save_folder rebuttal/5_path64
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/6.png --num_paths 64 --save_folder rebuttal/6_path64
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/7.png --num_paths 64 --save_folder rebuttal/7_path64
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/8.png --num_paths 64 --save_folder rebuttal/8_path64
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/9.png --num_paths 64 --save_folder rebuttal/9_path64
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/10.png --num_paths 64 --save_folder rebuttal/10_path64
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/11.png --num_paths 64 --save_folder rebuttal/11_path64
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/12.png --num_paths 64 --save_folder rebuttal/12_path64
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/13.png --num_paths 64 --save_folder rebuttal/13_path64
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/14.png --num_paths 64 --save_folder rebuttal/14_path64
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/15.png --num_paths 64 --save_folder rebuttal/15_path64
