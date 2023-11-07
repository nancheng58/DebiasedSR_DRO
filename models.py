import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.init import xavier_uniform_, xavier_normal_


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class SASRec(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder.
        ntoken: the size of vocab.--> token embedding
        max_length: the max length of the user-item-interaction sequence.--> position-embedding
        nfeature: the num of the feature field.--> feature embedding
        naspect: user aspect, item aspect and interaction aspect.--> aspect embedding
        user_basis: the dim of user latent factor from the NMF (may be changed for different datasets)
        item_basis: the dim of item latent factor from the NMF (may be changed for different datasets)
    """

    def __init__(self, args):
        super(SASRec, self).__init__()
        # ntoken, max_length, max_gen_step, d_model = 256, nhead = 8, hidden_size = 1024, nlayers = 4,
        # dropout = 0.1, teacher_forcing_ratio = 0.5
        d_model = args.hidden_size
        self.args = args
        self.max_length = args.max_length
        self.item_size = args.item_size
        self.norm = nn.LayerNorm(d_model, eps=1e-5)
        self.encoder_layers = TransformerEncoderLayer(d_model, args.nhead, d_model, args.attention_probs_dropout_prob)
        self.transformer_encoder = TransformerEncoder(self.encoder_layers, args.nlayers, self.norm)
        # 这里sequence生成embedding-matrix
        self.token_emb = nn.Embedding(args.item_size, d_model)
        self.position_emb = nn.Embedding(self.max_length, d_model)

        self.LayerNorm = LayerNorm(d_model, eps=1e-12)
        self.dropout = nn.Dropout(args.attention_probs_dropout_prob)
        self.pad_token = 0
        self.d_model = d_model
        self.log_sigmoid = nn.LogSigmoid()
        self.log_softmax = nn.LogSoftmax()
        self.fc = nn.Linear(d_model, args.item_size)
        self.apply(self.init_weights)

    def embedding(self, seq_token, seq_pos):  # embedding layer
        """
        To get the embeddings of the given sequence.
        :param seq_token: [B, L]
        :param seq_pos: [B, L]
        :return:
        """
        src_token_emb = self.token_emb(seq_token)  # [B, L, H]
        # print('src_token_emb-shape:',src_token_emb.shape)
        src_token_emb = src_token_emb.transpose(1, 0)  # [L,B,H]
        # print('token-emb[L,B,H]:',src_token_emb.shape)
        src_pos_emb = self.position_emb(seq_pos).transpose(1,
                                                           0)  # [L,B,H]     position embedding (rather than position encoding)
        # print('global-pos-emb[L,B,H]:',src_global_pos_emb.shape)
        # print(src_token_emb.shape, src_pos_emb.shape)

        src_embeddings = src_token_emb + src_pos_emb  # [L, B, H]
        # print('embs[L, B, H]:', src_embeddings.shape)

        src_embeddings = self.LayerNorm(src_embeddings)
        src_embeddings = self.dropout(src_embeddings)

        return src_embeddings

    def forward(self, seq_token, seq_pos):
        '''
        same ss sasrec
        :param seq_token: [B, L]
        :param seq_pos: [B, L]
        :return:
        '''
        padding_mask = (seq_token == self.pad_token).long()
        src_embeddings = self.embedding(seq_token, seq_pos)  # [L, B, H]

        # 第一步: 向右去预测
        output = self.transformer_encoder(src_embeddings,
                                          src_key_padding_mask=padding_mask.bool())  # [L, B, H] 这里一定要用bool类型
        # output = self.dense(output)  # 在transformer后面加上一层MLP，因为transformer-encoder最后一层是dropout
        # print('right-transformer-output-shape:', output.shape)  # [L,B,H]
        output = output.transpose(1, 0)[:, -1, :]
        return output

    def predict(self, seq_token, seq_pos):
        src_embeddings = self.embedding(seq_token, seq_pos)  # [L, B, H]
        batch_size = src_embeddings.shape[1]
        padding_mask = (seq_token == self.pad_token).long()
        # 向右预测
        output = self.transformer_encoder(src_embeddings,
                                          src_key_padding_mask=padding_mask.bool())
        output = output.transpose(1, 0)
        return output[:, -1, :]

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class GRU4Rec(nn.Module):
    def __init__(self, args):
        super(GRU4Rec, self).__init__()
        gru_layers = 1
        d_model = args.hidden_size
        self.max_length = args.max_length
        self.token_emb = nn.Embedding(args.item_size, d_model)
        self.dropout_prob = args.attention_probs_dropout_prob
        self.token_emb_dropout = nn.Dropout(self.dropout_prob)
        # nn.init.normal_(self.token_emb.weight, 0, 0.01)
        self.gru = nn.GRU(
            input_size=args.hidden_size,
            hidden_size=args.hidden_size,
            num_layers=gru_layers,
            batch_first=True,
            bias=False
        )
        self.fc = nn.Linear(d_model, args.item_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight)
        elif isinstance(module, nn.GRU):
            xavier_uniform_(module.weight_hh_l0)
            xavier_uniform_(module.weight_ih_l0)

    # def forward(self, seq_token, seq_pos):
    #     seq_embeddings = self.token_emb(seq_token)
    #     seq_embeddings_dropout = self.token_emb_dropout(seq_embeddings)
    #     # seq_pos = seq_pos.to('cpu')
    #     # emb_packed = torch.nn.utils.rnn.pack_padded_sequence(seq_embeddings, seq_pos, batch_first=True, enforce_sorted=False)
    #     gru_output, hidden = self.gru(seq_embeddings)
    #     # output = hidden.view(-1, hidden.shape[2])
    #     output = self.gather_indexes(gru_output, seq_pos - 1)
    #     # output = hidden[-1].view(-1, hidden[-1].shape[1])
    #     # output = self.fc(hidden)
    #     return output
    def forward(self, seq_token, seq_pos):
        # Supervised Head
        seq_embeddings = self.token_emb(seq_token)
        seq_pos = seq_pos.to('cpu')
        emb_packed = torch.nn.utils.rnn.pack_padded_sequence(seq_embeddings, seq_pos, batch_first=True,
                                                             enforce_sorted=False)
        emb_packed, hidden = self.gru(emb_packed)
        output = hidden[-1].view(-1, hidden[-1].shape[1])
        # output = hidden.view(-1, hidden.shape[2])
        # output = self.fc(hidden)
        return output

    # def predict(self, seq_token, seq_pos):
    #     seq_embeddings = self.token_emb(seq_token)
    #     seq_embeddings_dropout = self.token_emb_dropout(seq_embeddings)
    #     # seq_pos = seq_pos.to('cpu')
    #     # emb_packed = torch.nn.utils.rnn.pack_padded_sequence(seq_embeddings, seq_pos, batch_first=True, enforce_sorted=False)
    #     gru_output, hidden = self.gru(seq_embeddings)
    #     # output = hidden.view(-1, hidden.shape[2])
    #     output = self.gather_indexes(gru_output, seq_pos - 1)
    #     # output = hidden[-1].view(-1, hidden[-1].shape[1])
    #     # output = self.fc(hidden)
    #     return output

    def predict(self, seq_token, seq_pos):
        # Supervised Head
        seq_embeddings = self.token_emb(seq_token)
        seq_pos = seq_pos.to('cpu')
        emb_packed = torch.nn.utils.rnn.pack_padded_sequence(seq_embeddings, seq_pos, batch_first=True,
                                                             enforce_sorted=False)
        emb_packed, hidden = self.gru(emb_packed)
        output = hidden[-1].view(-1, hidden[-1].shape[1])
        # output = hidden.view(-1, hidden.shape[2])
        # output = self.fc(hidden)
        return output

    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

class Pop(nn.Module):
    r"""Pop is a fundamental model that always recommend the most popular item."""

    def __init__(self, args):
        super(Pop, self).__init__()
        self.item_cnt = torch.nn.Parameter(torch.zeros(args.item_size, dtype=torch.long), requires_grad=False)
        self.max_cnt = torch.nn.Parameter(torch.zeros([]))
        # self.item_list = args.item_counter_list
        # self.args = args
        self.fake_loss = torch.nn.Parameter(torch.zeros(1))
        self.other_parameter_name = ["item_cnt", "max_cnt"]

    def forward(self):
        pass

    def calculate_pop(self, args):
        self.item_cnt = Parameter(torch.tensor(args.item_counter_list, dtype=torch.long, requires_grad=False),requires_grad=False)
        self.max_cnt = Parameter(torch.max(self.item_cnt, dim=0)[0], requires_grad=False)
        return torch.nn.Parameter(torch.zeros(1))

    def predict(self, batchsize):
        result = self.item_cnt.to(torch.float64) / self.max_cnt.to(torch.float64)
        result = torch.repeat_interleave(result.unsqueeze(0), batchsize, dim=0)
        return result

    # def full_sort_predict(self, interaction):
    #     batch_user_num = interaction[self.USER_ID].shape[0]
    #     result = self.item_cnt.to(torch.float64) / self.max_cnt.to(torch.float64)
    #     result = torch.repeat_interleave(result.unsqueeze(0), batch_user_num, dim=0)
    #     return result.view(-1)

class ExposureModel(nn.Module):
    def __init__(self, args):
        super(ExposureModel, self).__init__()
        self.SASRecModel = SASRec(args)
        self.GRU4RecModel = GRU4Rec(args)
        self.PopModel = Pop(args)
        self.model_type = args.exposure_model_name

    def forward(self, sasrec_input_ids, seq_position, gru_input_ids, seq_len):
        if self.model_type == "mix":
            sas_out = self.SASRecModel.forward(sasrec_input_ids, seq_position)
            gru_out = self.GRU4RecModel.forward(gru_input_ids, seq_len)
            return sas_out, gru_out
        elif "sas" in self.model_type.lower():
            return self.SASRecModel.forward(sasrec_input_ids, seq_position)
        else:
            return self.GRU4RecModel.forward(gru_input_ids, seq_len)

    def dot_product(self, model, seq_out):
        # [item_num hidden_size]
        test_item_emb = model.token_emb.weight
        # [batch hidden_size]
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred

    def predict(self, sasrec_input_ids, seq_position, gru_input_ids, seq_len):
        batch_size = sasrec_input_ids.shape[0]
        if self.model_type == "mix":
            sas_out = self.SASRecModel.forward(sasrec_input_ids, seq_position)
            gru_out = self.GRU4RecModel.forward(gru_input_ids, seq_len)
            pop_out = self.PopModel.predict(batch_size).to(sas_out.device)
            # post softmax
            # sas_pred = self.dot_product(self.SASRecModel, sas_out)
            # gru_pred = self.dot_product(self.GRU4RecModel, gru_out)
            # pop_pred = pop_out * 0.3
            # return torch.softmax(sas_pred + gru_pred + pop_pred, 1)
            # prior softmax
            sas_pred = torch.softmax(self.dot_product(self.SASRecModel, sas_out), 1)
            gru_pred = torch.softmax(self.dot_product(self.GRU4RecModel, gru_out), 1)
            pop_pred = torch.softmax(pop_out, 1) * 0.3
            tot_out = sas_pred + gru_pred + pop_pred
            tot_pred = torch.sum(tot_out, dim=1).unsqueeze(-1).repeat(1, tot_out.shape[1])
            pred = torch.div(tot_out, tot_pred)
            return pred

        elif "sas" in self.model_type.lower():
            sas_out = self.SASRecModel.forward(sasrec_input_ids, seq_position)
            return torch.softmax(self.dot_product(self.SASRecModel, sas_out), 1)
        else:
            gru_out = self.GRU4RecModel.forward(gru_input_ids, seq_len)
            return torch.softmax(self.dot_product(self.GRU4RecModel, gru_out), 1)

    def pop_calcu(self):
        pass