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



python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/1.png --num_paths 3 --save_folder rebuttal/1_path3
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/2.png --num_paths 3 --save_folder rebuttal/2_path3
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/3.png --num_paths 3 --save_folder rebuttal/3_path3
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/4.png --num_paths 3 --save_folder rebuttal/4_path3
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/5.png --num_paths 3 --save_folder rebuttal/5_path3
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/6.png --num_paths 3 --save_folder rebuttal/6_path3
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/7.png --num_paths 3 --save_folder rebuttal/7_path3
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/8.png --num_paths 3 --save_folder rebuttal/8_path3
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/9.png --num_paths 3 --save_folder rebuttal/9_path3
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/10.png --num_paths 3 --save_folder rebuttal/10_path3
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/11.png --num_paths 3 --save_folder rebuttal/11_path3
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/12.png --num_paths 3 --save_folder rebuttal/12_path3
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/13.png --num_paths 3 --save_folder rebuttal/13_path3
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/14.png --num_paths 3 --save_folder rebuttal/14_path3
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/15.png --num_paths 3 --save_folder rebuttal/15_path3



python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/1.png --num_paths 4 --save_folder rebuttal/1_path4
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/2.png --num_paths 4 --save_folder rebuttal/2_path4
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/3.png --num_paths 4 --save_folder rebuttal/3_path4
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/4.png --num_paths 4 --save_folder rebuttal/4_path4
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/5.png --num_paths 4 --save_folder rebuttal/5_path4
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/6.png --num_paths 4 --save_folder rebuttal/6_path4
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/7.png --num_paths 4 --save_folder rebuttal/7_path4
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/8.png --num_paths 4 --save_folder rebuttal/8_path4
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/9.png --num_paths 4 --save_folder rebuttal/9_path4
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/10.png --num_paths 4 --save_folder rebuttal/10_path4
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/11.png --num_paths 4 --save_folder rebuttal/11_path4
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/12.png --num_paths 4 --save_folder rebuttal/12_path4
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/13.png --num_paths 4 --save_folder rebuttal/13_path4
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/14.png --num_paths 4 --save_folder rebuttal/14_path4
python painterly_rendering_ablation.py ../../Layerwise/data/rebuttal/15.png --num_paths 4 --save_folder rebuttal/15_path4
