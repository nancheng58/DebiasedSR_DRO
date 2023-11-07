#!/bin/bash
#SBATCH -e result/ZhihuRec-sas.err
#SBATCH -o result/ZhihuRec-sas.out
#SBATCH -J DRO_recommender
#SBATCH --partition= [your_partition]
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=999:00:00

conda activate [your_env]
python run_full.py --data_name='Zhihurec' --ckp=0 --batch_size=512 --epochs=400 --hidden_size=64 --lr=0.005 --dro_reg=1000 --debias_evaluation_k=0.1 --use_exposure_data=1 --model_name="SASRec" --exposure_model_name="mix"