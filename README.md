
# DebiasedSR_DRO
This is a PyTorch implementation for the [paper](https://arxiv.org/abs/2312.07036) accepted by WSDM 2024 (oral presentation):
> Jiyuan Yang, Yue Ding, Yidan Wang, Pengjie Ren, Zhumin Chen, Fei Cai, Rui Zhang, Jun Ma, Zhaochun Ren, Xin Xin. Debiasing Sequential Recommenders through Distributionally Robust Optimization over System Exposure.


# Overview

In this paper, we propose to debias sequential recommenders through Distributionally Robust Optimization (DRO) over system exposure data.
The key idea is to utilize DRO to optimize the worst-case error over an uncertainty set to safeguard the model against distributional discrepancy caused by the exposure bias. 
The main challenge to apply DRO for exposure debiasing in sequential recommendation lies in how to construct the uncertainty set and avoid the overestimation of user preference on biased samples. Moreover, since the test set could also be affected by the exposure bias, how to evaluate the debiasing effect is also an open question.
To this end, we first introduce an exposure simulator trained upon the system exposure data to calculate the exposure distribution, which is then regarded as the nominal distribution to construct the uncertainty set of DRO. Then, we introduce a penalty to items with high exposure probability to avoid the overestimation of user preference for biased samples. 
Finally, we design a debiased self-normalized inverse propensity score (SNIPS) evaluator for evaluating the debiasing effect on the biased offline test set. 
We conduct extensive experiments on two real-world datasets to verify the effectiveness of the proposed methods. Experimental results demonstrate the superior exposure debiasing performance of the proposed methods. 
![http-bw](assets/DRO.svg)

## Requirements
```
torch==1.1.0
numpy==1.19.1
scipy==1.5.2
tqdm==4.48.2
```

## Datasets
The default dataset setting of code is ZhihuRec and run
```bash
python data/ZhihuRec/data_process_ZhihuRec.py
```
to preprocess the dataset.

For the Tenrec dataset, you need to acquire the licence, and more details can be found in this [link](https://github.com/yuangh-x/2022-NIPS-Tenrec). 
After that, put the 'QB-video.csv' into 'data/Tenrec' and run
```bash
python data/Tenrec/data_process_Tenrec.py
```

## Recommender
To reproduce results of recomender,
```bash
python run_full.py
```

## Exposure simulator and Evaluation simulator
To access the Exposure Simulator and Evaluation Simulator,
```bash
python exposure.py
```
```bash
python evaluation.py
```

## Slurm script

Besides, we provide the slurm execute script.

More details about slurm usage can be found on this link: https://slurm.schedmd.com/documentation.html

*[Note: please modify  "conda activate envname" to your environment]*


# Cite
The final version of this paper is coming soon.

## Reference
The code is implemented based on [S3-Rec](https://github.com/RUCAIBox/CIKM2020-S3Rec).
