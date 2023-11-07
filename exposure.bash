#!/bin/bash
#SBATCH -e result/Exposure-ZhihuRec-mix.err
#SBATCH -o result/Exposure-ZhihuRec-mix.out
#SBATCH -J Exposure
#SBATCH --partition= [your_partition]
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=999:00:00


conda activate [your_env]
python run_exposure.py --data_name='ZhihuRec' --batch_size=512 --epochs=200 --hidden_size=64 --lr=0.005 --exposure_model_name='mix'
