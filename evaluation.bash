#!/bin/bash
#SBATCH -e result/Evaluation-ZhihuRec-mix.err
#SBATCH -o result/Evaluation-ZhihuRec-mix.out
#SBATCH -J Evaluation
#SBATCH --partition= [your_partition]
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=999:00:00


conda activate [your_env]
python run_evaluation.py --data_name='ZhihuRec' --batch_size=512 --epochs=200 --hidden_size=64 --lr=0.005 --exposure_model_name='mix'
