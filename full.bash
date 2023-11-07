#!/bin/bash
#SBATCH -e result/Tenrec-sas.err
#SBATCH -o result/Tenrec-sas.out
#SBATCH -J 1_DRO
#SBATCH --partition=si
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=999:00:00

conda activate torch1.10 
python run_full.py --data_name='Tenrec' --ckp=0 --batch_size=512 --epochs=400 --hidden_size=64 --lr=0.005 --dro_reg=1000 --debias_evaluation_k=0.1 --use_exposure_data=1 --model_name="SASRec" --exposure_model_name="mix"