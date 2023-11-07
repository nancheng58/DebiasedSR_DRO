# -*- coding: utf-8 -*-
import datetime
import os
import numpy as np
import random
import torch
import argparse

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from datasets import Dataset
from trainers import Trainer
from models import SASRec, GRU4Rec, ExposureModel
from utils import EarlyStopping, check_path, set_seed, get_user_seqs, generate_rating_matrix


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default='./data/', type=str)
    parser.add_argument('--output_dir', default='output/', type=str)
    parser.add_argument('--data_name', default='ZhihuRec', type=str)
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--ckp', default=0, type=int, help="pretrain epochs 10, 20, 30...")
    parser.add_argument('--use_exposure_data', default=1, type=int)

    # model args
    parser.add_argument("--model_name", default="SASRec", type=str)
    parser.add_argument("--hidden_size", type=int, default=64, help="hidden size of transformer model")
    parser.add_argument("--nlayers", type=int, default=2, help="number of layers")
    parser.add_argument('--nhead', default=2, type=int)
    parser.add_argument('--hidden_act', default="gelu", type=str)  # gelu relu
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.5, help="attention dropout p")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.5, help="hidden dropout p")
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument('--max_length', default=50, type=int)
    parser.add_argument('--exposure_max_length', default=200, type=int)
    parser.add_argument('--dro_reg', default=1, type=float, help="-1 IPS, -2 IPS-C")
    parser.add_argument('--exposure_model_name', default="mix", type=str, help="SASrec, GRU4rec, or mix")

    # train and test args
    parser.add_argument("--lr", type=float, default=0.005, help="learning rate of adam")
    parser.add_argument("--batch_size", type=int, default=512, help="number of batch_size")
    parser.add_argument("--epochs", type=int, default=400, help="number of epochs")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument('--debias_evaluation_k', default=0.1, type=float)
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")
    parser.add_argument("--n_warmup_steps", type=int, default=4000, help="warmup step")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")
    args = parser.parse_args()

    set_seed(args.seed)
    check_path(args.output_dir)

    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = True
    # save model args
    time_stamp = datetime.datetime.now()

    item_size = 0
    args.data_file = args.data_dir + args.data_name
    _, _, item_counter = get_user_seqs(args.data_file + "/ori_Train.txt")
    train_data, max_item, _ = get_user_seqs(args.data_file + "/Train.txt")
    item_size = max(item_size, max_item)
    niche_user = int(item_size * 0.2)
    niche_set = set([user[0] for user in item_counter.most_common()[
                                         -niche_user - 1: -1]])
    valid_data, max_item, _ = get_user_seqs(args.data_file + "/Valid.txt")
    item_size = max(item_size, max_item)
    test_data, max_item, _ = get_user_seqs(args.data_file + "/Test.txt")
    item_size = max(item_size, max_item)
    args.item_size = item_size + 2
    valid_matrix = generate_rating_matrix(valid_data, args.item_size)
    test_matrix = generate_rating_matrix(test_data, args.item_size)
    args.train_matrix = valid_matrix
    args_str = f'{args.model_name}-{args.data_name}-{args.dro_reg}-Exposure{args.use_exposure_data}'
    args.log_file = os.path.join(args.output_dir, args_str + '.txt')
    print(str(args))
    with open(args.log_file, 'a') as f:
        f.write(str(time_stamp))
        f.write(str(args) + '\n')
    args_str = args_str + f'-{time_stamp}'
    # save model
    checkpoint = args_str + '.pt'
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)

    train_dataset = Dataset(args, train_data, model_name=args.model_name)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

    eval_dataset = Dataset(args, valid_data, model_name=args.model_name)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)

    test_dataset = Dataset(args, test_data, model_name=args.model_name)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size)

    model = SASRec(args=args) if args.model_name == "SASRec" else GRU4Rec(args=args)
    exposure_args = args
    exposure_args.max_length = args.exposure_max_length
    exposure_model = ExposureModel(args=exposure_args)
    evaluation_model = ExposureModel(args=exposure_args)  # max length of exposure model is same as evaluation model
    trainer = Trainer(model, train_dataloader, eval_dataloader,
                      test_dataloader, args, exposure_model, evaluation_model, niche_set)

    if args.debias_evaluation_k > 0:
        pretrained_evaluation_path = os.path.join(args.output_dir, f'mix-{args.data_name}-Evaluation.pt')
        try:
            trainer.load_evaluation(pretrained_evaluation_path)
            test_data, max_item, _ = get_user_seqs(args.data_file + "/Test.txt")
            test_dataset = Dataset(args, test_data, model_name="SASRec")
            trainer.evaluation_pred(
                DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=args.batch_size))
            print(f'Load pre-trained evaluation model from {pretrained_evaluation_path}!')
        except FileNotFoundError:
            print(f'{pretrained_evaluation_path} Not found the pre-trained evaluation model !')
    if args.do_eval:
        trainer.load(args.checkpoint_path)
        print(f'Load model from {args.checkpoint_path} for test!')
        scores, result_info = trainer.test(0, full_sort=True)

    else:
        if args.use_exposure_data > 0 or args.dro_reg < 0:  # the cal of IPS needs the exposure model
            pretrained_exposure_path = os.path.join(args.output_dir, f'{args.exposure_model_name}-{args.data_name}-Exposure.pt')
            try:
                trainer.load_exposure(pretrained_exposure_path)
                print(f'Load pre-trained exposure model from {pretrained_exposure_path}!')

            except FileNotFoundError:
                print(f'{pretrained_exposure_path} Not found the pre-trained exposure model !')

        early_stopping = EarlyStopping(args.checkpoint_path, patience=30, verbose=True)

        for epoch in range(args.epochs):
            trainer.train(epoch)
            # evaluate on NDCG@20
            scores, _ = trainer.valid(epoch, full_sort=True)
            early_stopping(np.array(scores[-1:]), trainer.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        print('---------------Change to test_rating_matrix!-------------------')
        # load the best model
        args.train_matrix = test_matrix
        trainer.model.load_state_dict(torch.load(args.checkpoint_path))
        scores, result_info = trainer.test(0, full_sort=True)

    print(args_str)
    print(result_info)
    with open(args.log_file, 'a') as f:
        f.write(args_str + '\n')
        f.write(result_info + '\n')


main()
