# -*- coding: utf-8 -*-
import os
import numpy as np
import random
import torch
import argparse

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from datasets import Dataset
from trainers import ExposureTrainer
from models import ExposureModel
from utils import EarlyStopping, check_path, set_seed, get_user_seqs, generate_rating_matrix


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default='./data/', type=str)
    parser.add_argument('--output_dir', default='output/', type=str)
    parser.add_argument('--data_name', default='ZhihuRec', type=str)
    parser.add_argument('--do_eval', action='store_true')

    # model args
    parser.add_argument('--exposure_model_name', default="mix", type=str, help="SASRec, GRU4Rec, or mix")
    parser.add_argument("--hidden_size", type=int, default=64, help="hidden size of transformer model")
    parser.add_argument("--nlayers", type=int, default=2, help="number of layers")
    parser.add_argument('--nhead', default=2, type=int)
    parser.add_argument('--hidden_act', default="gelu", type=str)  # gelu relu
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.5, help="attention dropout p")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.5, help="hidden dropout p")
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument('--max_length', default=200, type=int)
    parser.add_argument('--dro_reg', default=0.0, type=float)
    parser.add_argument('--debias_evaluation_k', default=0, type=float)

    # train args
    parser.add_argument("--lr", type=float, default=0.005, help="learning rate of adam")
    parser.add_argument("--batch_size", type=int, default=512, help="number of batch_size")
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=42, type=int)
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
    item_size = 0
    args.data_file = args.data_dir + args.data_name
    _, _, item_counter = get_user_seqs(args.data_file + "/ori_Evaluation_Train.txt")
    train_data, max_item, _ = get_user_seqs(args.data_file + "/Evaluation_Train.txt")  # Exposure_Train
    item_size = max(item_size, max_item)
    valid_data, max_item, _ = get_user_seqs(args.data_file + "/Evaluation_Valid.txt")
    item_size = max(item_size, max_item)
    test_data, max_item, _ = get_user_seqs(args.data_file + "/Evaluation_Test.txt")
    item_size = max(item_size, max_item)
    args.item_size = item_size + 2
    valid_matrix = generate_rating_matrix(valid_data, args.item_size)
    test_matrix = generate_rating_matrix(test_data, args.item_size)
    args.train_matrix = valid_matrix
    item_counter_list = [item_counter.get(i) for i in range(0, args.item_size)]
    for i in range(0, args.item_size):
        if item_counter_list[i] is None:
            item_counter_list[i] = 0
    args.item_counter_list = item_counter_list
    # save model args
    args_str = f'{args.exposure_model_name}-{args.data_name}-Evaluation'
    args.log_file = os.path.join(args.output_dir, args_str + '.txt')
    print(str(args))
    with open(args.log_file, 'a') as f:
        f.write(str(args) + '\n')

    # save model
    checkpoint = args_str + '.pt'
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)

    train_dataset = Dataset(args, train_data)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

    eval_dataset = Dataset(args, valid_data)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)

    test_dataset = Dataset(args, test_data)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size)

    model = ExposureModel(args=args)

    trainer = ExposureTrainer(model, train_dataloader, eval_dataloader,
                              test_dataloader, args)

    if args.do_eval:
        trainer.load(args.checkpoint_path)
        print(f'Load model from {args.checkpoint_path} for test!')
        scores, result_info = trainer.test(0, full_sort=True)

    else:

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
