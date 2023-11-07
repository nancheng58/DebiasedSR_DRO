import torch
from torch.utils.data import Dataset

from utils import neg_sample


class Dataset(Dataset):

    def __init__(self, args, user_seq, model_name="SASRec"):
        self.args = args
        self.user_seq = user_seq
        self.model_name = model_name
        self.max_len = args.max_length

    def __getitem__(self, index):

        user_id = index
        items = self.user_seq[index]
        # [0, 1, 2, 3, 4, 5, 6]
        # for train, valid, and test
        # input_ids [0, 1, 2, 3, 4, 5]
        # answer [6]

        input_ids = items[:-1]
        seq_length = len(input_ids)
        answer = items[-1]
        target_neg = (neg_sample(answer, self.args.item_size))
        pad_len = self.max_len - len(input_ids)
        seq_position = [0] * pad_len + list(range(0, len(input_ids)))
        gru_input_ids = input_ids + [0] * pad_len
        gru_input_ids = gru_input_ids[:self.max_len]
        input_ids = [0] * pad_len + input_ids
        input_ids = input_ids[-self.max_len:]
        assert len(input_ids) == self.max_len

        cur_tensors = (
            torch.tensor(user_id, dtype=torch.long),  # user_id for testing
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(answer, dtype=torch.long),
            torch.tensor(seq_position, dtype=torch.long),
            torch.tensor(target_neg, dtype=torch.long),
            torch.tensor(gru_input_ids, dtype=torch.long),
            torch.tensor(seq_length, dtype=torch.long),
        )
        return cur_tensors

    def __len__(self):
        return len(self.user_seq)
