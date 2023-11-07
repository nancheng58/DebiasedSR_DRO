#!/bin/bash
#SBATCH -e result/Exposure-zhihurec-lr=0.005-SASRec.err
#SBATCH -o result/Exposure-zhihurec-lr=0.005-SASRec.out
#SBATCH -J Exposure
#SBATCH --partition=si
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=999:00:00


conda activate torch1.10 
python run_exposure.py --data_name='zhihurec' --batch_size=512 --epochs=200 --hidden_size=64 --lr=0.005 --exposure_model_name='SASRec'
