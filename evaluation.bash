#!/bin/bash
#SBATCH -e result/Evaluation-zhihurec-sas_with-DRO-lr=0.005-mix.err
#SBATCH -o result/Evaluation-zhihurec-sas_with-DRO-lr=0.005-mix.out
#SBATCH -J Evaluation
#SBATCH --partition=si
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=999:00:00


conda activate torch1.10 
python run_evaluation.py --data_name='zhihurec' --batch_size=512 --epochs=200 --hidden_size=64 --lr=0.005 --exposure_model_name='mix'
