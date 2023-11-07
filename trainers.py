# -*- coding: utf-8 -*-
import numpy as np
import tqdm
import torch
from torch.optim import Adam
from optim import ScheduledOptim
from utils import recall_at_k, ndcg_k, check_gpu_capability, coverage_at_k, apt_at_k


class Trainer:
    def __init__(self, model, train_dataloader,
                 eval_dataloader,
                 test_dataloader, args, exposure_model=None, evaluation_model=None, niche_set=None):

        if niche_set is None:
            niche_set = set()
        self.args = args
        self.cuda_condition = check_gpu_capability() and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")

        self.model = model
        self.exposure_model = exposure_model
        self.evaluation_model = evaluation_model
        if self.cuda_condition:
            self.model.cuda()
            if exposure_model is not None:
                self.exposure_model.cuda()
            if evaluation_model is not None:
                self.evaluation_model.cuda()
        # Setting the train, valid, and test data loader
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader
        self.optim = ScheduledOptim(Adam(model.parameters(), betas=(args.adam_beta1, args.adam_beta2), eps=1e-09,
                                         weight_decay=args.weight_decay), n_warmup_steps=args.n_warmup_steps,
                                    init_lr=args.lr)
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
        self.dro_reg = self.args.dro_reg
        if exposure_model is None:
            self.use_exposure_data = 0
        else:
            self.use_exposure_data = self.args.use_exposure_data
        self.eval_pred = None
        self.niche_set = niche_set

    def train(self, epoch):
        return self.iteration(epoch, self.train_dataloader)

    def valid(self, epoch, full_sort=False):
        return self.iteration(epoch, self.eval_dataloader, full_sort, train=False)

    def test(self, epoch, full_sort=False):
        return self.iteration(epoch, self.test_dataloader, full_sort, train=False)

    def iteration(self, epoch, dataloader, full_sort=False, train=True):

        str_code = "train" if train else "test"

        # Setting the tqdm progress bar
        rec_data_iter = tqdm.tqdm(enumerate(dataloader),
                                  desc="Recommendation EP_%s:%d" % (str_code, epoch),
                                  total=len(dataloader),
                                  bar_format="{l_bar}{r_bar}")
        if train:
            self.model.train()
            if self.exposure_model is not None:
                self.exposure_model.eval()
            if self.evaluation_model is not None:
                self.evaluation_model.eval()
            avg_loss = 0.0
            rec_avg_loss = 0.0
            dro_avg_loss = 0.0
            for i, batch in rec_data_iter:
                # 0. batch_data will be sent into the device(GPU or CPU)
                self.optim.zero_grad()
                batch = tuple(t.to(self.device) for t in batch)
                _, sasrec_input_ids, rec_answer, seq_position, target_neg, gru_input_ids, seq_len = batch
                if 'sas' in self.args.model_name.lower():
                    input_ids = sasrec_input_ids
                    input_position = seq_position
                else:
                    input_ids = gru_input_ids
                    input_position = seq_len
                output = self.model(input_ids, input_position)
                rec_loss = self.cross_entropy(output, rec_answer, target_neg)
                if abs(self.dro_reg - 0.0) < 1e-5:  # dro_reg = 0, same as SASRec
                    loss = rec_loss
                elif self.dro_reg > 0:  # dro_reg > 0, rec_loss + dro_loss
                    if self.use_exposure_data == 0:  # adopt uniform distribution
                        dro_loss = self.dro_uniform(self.model.fc(output), rec_answer)
                    else:
                        dro_loss = self.dro(input_ids, seq_position, gru_input_ids, seq_len, self.model.fc(output),
                                            rec_answer)
                    loss = rec_loss + self.dro_reg * dro_loss
                    dro_avg_loss += dro_loss.item()
                else:  # dro_reg < 0 , IPS
                    loss = self.ips(input_ids, seq_position, gru_input_ids, seq_len, output, rec_answer,
                                    target_neg, -self.dro_reg)
                loss.backward()
                self.optim.step_and_update_lr()
                avg_loss += loss.item()
                rec_avg_loss += rec_loss.item()
            post_fix = {
                "epoch": epoch,
                "loss": '{:.4f}'.format(avg_loss / len(rec_data_iter)),
                "rec_loss": '{:.4f}'.format(rec_avg_loss / len(rec_data_iter)),
                "dro_loss": '{:.4f}'.format(dro_avg_loss / len(rec_data_iter)),
            }
            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix))

            with open(self.args.log_file, 'a') as f:
                f.write(str(post_fix) + '\n')

            '''Debug'''
            return rec_avg_loss / len(rec_data_iter)
        # valid and test
        else:
            self.model.eval()
            if self.exposure_model is not None:
                self.exposure_model.eval()
            if self.evaluation_model is not None:
                self.evaluation_model.eval()
            pred_list = None
            answer_list = None
            for i, batch in rec_data_iter:
                # 0. batch_data will be sent into the device(GPU or cpu)
                batch = tuple(t.to(self.device) for t in batch)
                user_ids, sasrec_input_ids, answers, seq_position, target_neg, gru_input_ids, seq_len = batch
                if 'sas' in self.args.model_name.lower():
                    input_ids = sasrec_input_ids
                    input_position = seq_position
                else:
                    input_ids = gru_input_ids
                    input_position = seq_len
                recommend_output = self.model.predict(input_ids, input_position)
                rating_pred = self.predict_full(recommend_output)  # dot product
                rating_pred = rating_pred.cpu().data.numpy().copy()
                batch_user_index = user_ids.cpu().numpy()
                rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0
                ind = np.argpartition(rating_pred, -20)[:, -20:]
                arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                if i == 0:
                    pred_list = batch_pred_list
                    answer_list = answers.cpu().data
                else:
                    pred_list = np.append(pred_list, batch_pred_list, axis=0)
                    answer_list = np.append(answer_list, answers.cpu().data, axis=0)
            answer_list = answer_list.tolist()
            answer_list = [[answer] for answer in answer_list]
            pred_list = pred_list.tolist()
            return self.get_full_sort_score(epoch, answer_list, pred_list)
    def cross_entropy(self, seq_out, pos_ids, neg_ids):
        # [batch hidden_size]
        pos_emb = self.model.token_emb(pos_ids)
        neg_emb = self.model.token_emb(neg_ids)
        # [batch hidden_size]
        pos = pos_emb.view(-1, pos_emb.size(1))
        neg = neg_emb.view(-1, neg_emb.size(1))
        seq_emb = seq_out.view(-1, self.args.hidden_size)  # [batch hidden_size]
        pos_logits = torch.sum(pos * seq_emb, -1)  # [batch]
        neg_logits = torch.sum(neg * seq_emb, -1)
        istarget = (pos_ids > 0).view(pos_ids.size(0)).float()  # [batch]
        loss = torch.sum(
            - torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget -
            torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / torch.sum(istarget)

        return loss

    def ips(self, input_id, seq_position, gru_input_ids, seq_len, seq_out, pos_ids, neg_ids, ips_type=1):
        """
            ips_type:  1, IPS; 2, IPS-C; 3, RelMF
        """
        # shared calculation
        # [batch hidden_size]
        exp_rating_pred = self.exposure_pred(input_id, seq_position, gru_input_ids, seq_len)
        # exp_rating_pred = torch.pow(exp_rating_pred, 0.2)
        batch_size = exp_rating_pred.shape[0]
        item_size = exp_rating_pred.shape[1]
        uniform_pred = (1 / item_size) ** 0.2
        pos_emb = self.model.token_emb(pos_ids)
        neg_emb = self.model.token_emb(neg_ids)
        # [batch hidden_size]
        pos = pos_emb.view(-1, pos_emb.size(1))
        neg = neg_emb.view(-1, neg_emb.size(1))
        seq_emb = seq_out.view(-1, self.args.hidden_size)  # [batch hidden_size]
        pos_logits = torch.sum(pos * seq_emb, -1)  # [batch]
        neg_logits = torch.sum(neg * seq_emb, -1)
        istarget = (pos_ids > 0).view(pos_ids.size(0)).float()  # [batch]
        if ips_type == 1:  # IPS common version
            exp_rating_pred = torch.pow(exp_rating_pred, 0.2)
            pos_ps = exp_rating_pred.gather(dim=1, index=pos_ids.unsqueeze(-1)).squeeze(-1)
            neg_ps = torch.tensor(uniform_pred, device=self.device).repeat(batch_size)
            loss = torch.sum(
                - 1 / pos_ps * torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget
                - 1 / neg_ps * torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
            ) / torch.sum(istarget)
        elif ips_type == 2:  # IPS-C
            exp_rating_pred = torch.pow(exp_rating_pred, 0.2)
            pos_ps = exp_rating_pred.gather(dim=1, index=pos_ids.unsqueeze(-1)).squeeze(-1)
            # clip
            median, _ = torch.median(exp_rating_pred, dim=-1)
            pop = torch.clip(pos_ps, max=median)
            neg_ps = torch.tensor(uniform_pred, device=self.device).repeat(batch_size)
            loss = torch.sum(
                - 1 / pop * torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget
                - 1 / neg_ps * torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
            ) / torch.sum(istarget)
        elif ips_type == 3:  # RelMF
            exp_rating_pred = torch.pow(exp_rating_pred, 0.2)
            pos_ps = exp_rating_pred.gather(dim=1, index=pos_ids.unsqueeze(-1)).squeeze(-1)
            loss = torch.sum(
                - 1 / pos_ps * torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget
                - (1 - 1 / pos_ps) * torch.log(1 - torch.sigmoid(pos_logits) + 1e-24) * istarget -
                torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
            ) / torch.sum(istarget)
        else:
            raise NotImplementedError("Make sure 'ips_type' in [1, 2, 3]!")
        return loss

    def get_full_sort_score(self, epoch, answers, pred_list):
        recall, ndcg, coverage, apt, apt_p = [], [], [], [], []
        # epoch=0 means at testing stage now
        eval_pred = self.eval_pred if self.args.debias_evaluation_k > 0 else None
        for k in [5, 10, 20]:
            recall.append(recall_at_k(answers, pred_list, k, eval_pred))
            ndcg.append(ndcg_k(answers, pred_list, k, eval_pred))
            coverage.append(coverage_at_k(pred_list, k) / self.args.item_size)
            _apt, _apt_p = apt_at_k(answers, self.niche_set, pred_list, k, eval_pred)
            apt.append(_apt)
            apt_p.append(_apt_p)

        post_fix = {
            "Epoch": epoch,
            "Expo_eval: K = ": self.args.debias_evaluation_k,
            "Re@5": '{:.4f}'.format(recall[0]), "NDCG@5": '{:.4f}'.format(ndcg[0]),
            "Re@10": '{:.4f}'.format(recall[1]), "NDCG@10": '{:.4f}'.format(ndcg[1]),
            "Re@20": '{:.4f}'.format(recall[2]), "NDCG@20": '{:.4f}'.format(ndcg[2]),
            "coverage@5": '{:.4f}'.format(coverage[0]), "apt@5": '{:.4f}'.format(apt[0]),
            "coverage@10": '{:.4f}'.format(coverage[1]), "apt@10": '{:.4f}'.format(apt[1]),
            "coverage@20": '{:.4f}'.format(coverage[2]), "apt@20": '{:.4f}'.format(apt[2]),
            "apt_p@5": '{:.4f}'.format(apt_p[0]), "apt_p@10": '{:.4f}'.format(apt_p[1]),
            "apt_p@20": '{:.4f}'.format(apt_p[2]),
        }
        print(post_fix)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        return [recall[0], ndcg[0], recall[1], ndcg[1], recall[2], ndcg[2]], str(post_fix)

    def exposure_pred(self, sas_input_ids, seq_position, gru_input_ids, seq_len):
        with torch.no_grad():
            batch_size = sas_input_ids.shape[0]
            seq_padding = torch.tensor(0, device=self.device, requires_grad=False).repeat(
                self.args.exposure_max_length - self.model.max_length).unsqueeze(0).repeat(
                batch_size, 1)
            exposure_gru_input_ids = torch.cat((gru_input_ids, seq_padding), dim=1)
            exposure_seq_len = seq_len.detach()
            exposure_sas_input_ids = torch.cat((seq_padding, sas_input_ids), dim=1)
            exposure_seq_position = torch.cat((seq_padding, seq_position), dim=1)
            exp_rating_pred = self.exposure_model.predict(exposure_sas_input_ids, exposure_seq_position,
                                                          exposure_gru_input_ids, exposure_seq_len)
        return exp_rating_pred

    def evaluation_pred(self, ori_test_dataloader):
        dataloader = ori_test_dataloader
        str_code = "calculate_evaluation_pred"
        rec_data_iter = tqdm.tqdm(enumerate(dataloader),
                                  desc="Recommendation EP_%s" % (str_code),
                                  total=len(dataloader),
                                  bar_format="{l_bar}{r_bar}")
        user_tot = 0
        assert self.evaluation_model is not None
        self.evaluation_model.eval()
        for i, batch in rec_data_iter:
            # 0. batch_data will be sent into the device (GPU or cpu)
            batch = tuple(t.to(self.device) for t in batch)
            _, sasrec_input_ids, rec_answer, seq_position, target_neg, gru_input_ids, seq_len = batch
            batch_size = sasrec_input_ids.shape[0]
            user_tot += batch_size
            with torch.no_grad():
                seq_padding = torch.tensor(0, device=self.device, requires_grad=False).repeat(
                    self.args.exposure_max_length - self.model.max_length).unsqueeze(0).repeat(
                    batch_size, 1)
                evaluation_gru_input_ids = torch.cat((gru_input_ids, seq_padding), dim=1)
                evaluation_seq_len = seq_len.detach()
                evaluation_sasrec_input_ids = torch.cat((seq_padding, sasrec_input_ids), dim=1)
                evaluation_seq_position = torch.cat((seq_padding, seq_position), dim=1)
                exp_rating_pred = self.evaluation_model.predict(evaluation_sasrec_input_ids, evaluation_seq_position,
                                                                evaluation_gru_input_ids, evaluation_seq_len)
                exp_rating_pred = exp_rating_pred.sum(dim=0)
                if self.eval_pred is None:
                    self.eval_pred = exp_rating_pred.clone()
                else:
                    self.eval_pred += exp_rating_pred
        assert user_tot != 0
        self.eval_pred = torch.div(self.eval_pred, user_tot)
        self.eval_pred = torch.pow(self.eval_pred, self.args.debias_evaluation_k).tolist()

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name))

    def load_exposure(self, file_name):
        self.exposure_model.load_state_dict(torch.load(file_name))

    def load_evaluation(self, file_name):
        self.evaluation_model.load_state_dict(torch.load(file_name))

    def dro_uniform(self, output, rec_answer):  # DRO loss
        target_logits = output.gather(dim=1, index=rec_answer.unsqueeze(-1))
        item_size = output.shape[1]
        rating_pred = torch.log(float(1 / item_size) * torch.sum(torch.exp(output) + torch.exp(1 - target_logits),
                                                                 dim=1) + 1e-24).unsqueeze(-1)
        return rating_pred.mean()

    def dro(self, input_id, seq_position, gru_input_ids, seq_len, output, rec_answer):
        exp_rating_pred = self.exposure_pred(input_id, seq_position, gru_input_ids, seq_len)
        target_logits = output.gather(dim=1, index=rec_answer.unsqueeze(-1))
        exp_rating_target = exp_rating_pred.gather(dim=1, index=rec_answer.unsqueeze(-1))
        rating_pred = torch.log(
            torch.sum(exp_rating_pred * torch.exp(output) + exp_rating_target * (torch.exp(1 - target_logits)),
                      dim=1) + 1e-24).unsqueeze(-1)
        return rating_pred.mean()
    def predict_full(self, seq_out):
        # [item_num hidden_size]
        test_item_emb = self.model.token_emb.weight
        # [batch hidden_size ]
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred


'''
    Trainer of the exposure model.
'''
class ExposureTrainer(Trainer):
    def __init__(self, model,
                 train_dataloader,
                 eval_dataloader,
                 test_dataloader, args):
        super(ExposureTrainer, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader, args, None, None, None
        )
        model.PopModel.calculate_pop(args)

    def cross_entropy_loss(self, model, seq_out, pos_ids, neg_ids):
        # [batch hidden_size]
        pos_emb = model.token_emb(pos_ids)
        neg_emb = model.token_emb(neg_ids)
        # [batch hidden_size]
        pos = pos_emb.view(-1, pos_emb.size(1))
        neg = neg_emb.view(-1, neg_emb.size(1))
        seq_emb = seq_out.view(-1, self.args.hidden_size)  # [batch hidden_size]
        pos_logits = torch.sum(pos * seq_emb, -1)  # [batch]
        neg_logits = torch.sum(neg * seq_emb, -1)
        istarget = (pos_ids > 0).view(pos_ids.size(0)).float()  # [batch]
        loss = torch.sum(
            - torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget -
            torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / torch.sum(istarget)

        return loss

    def iteration(self, epoch, dataloader, full_sort=False, train=True):

        str_code = "train" if train else "test"

        # Setting the tqdm progress bar

        rec_data_iter = tqdm.tqdm(enumerate(dataloader),
                                  desc="Recommendation EP_%s:%d" % (str_code, epoch),
                                  total=len(dataloader),
                                  bar_format="{l_bar}{r_bar}")
        if train:
            self.model.train()
            avg_loss = 0.0
            rec_avg_sas_loss = 0.0
            rec_avg_gru_loss = 0.0
            for i, batch in rec_data_iter:
                # 0. batch_data will be sent into the device(GPU or CPU)
                self.optim.zero_grad()
                batch = tuple(t.to(self.device) for t in batch)
                _, sasrec_input_ids, rec_answer, seq_position, target_neg, gru_input_ids, seq_len = batch
                if self.args.exposure_model_name == "mix":
                    sas_output, gru_output = self.model(sasrec_input_ids, seq_position, gru_input_ids, seq_len)
                    rec_loss_sas = self.cross_entropy_loss(self.model.SASRecModel, sas_output, rec_answer, target_neg)
                    rec_loss_gru = self.cross_entropy_loss(self.model.GRU4RecModel, gru_output, rec_answer, target_neg)
                    loss = rec_loss_sas + rec_loss_gru
                    rec_avg_sas_loss += rec_loss_sas.item()
                    rec_avg_gru_loss += rec_loss_gru.item()
                elif self.args.exposure_model_name == "SASRec":
                    sas_output = self.model(sasrec_input_ids, seq_position, gru_input_ids, seq_len)
                    loss = self.cross_entropy_loss(self.model.SASRecModel, sas_output, rec_answer, target_neg)
                else:
                    gru_output = self.model(sasrec_input_ids, seq_position, gru_input_ids, seq_len)
                    loss = self.cross_entropy_loss(self.model.GRU4RecModel, gru_output, rec_answer, target_neg)
                loss.backward()
                self.optim.step_and_update_lr()
                avg_loss += loss.item()
            post_fix = {
                "epoch": epoch,
                "loss": '{:.4f}'.format(avg_loss / len(rec_data_iter)),
                "sas_loss": '{:.4f}'.format(rec_avg_sas_loss / len(rec_data_iter)),
                "gru_loss": '{:.4f}'.format(rec_avg_gru_loss / len(rec_data_iter)),
            }
            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix))

            with open(self.args.log_file, 'a') as f:
                f.write(str(post_fix) + '\n')

            return avg_loss / len(rec_data_iter)
        # valid and test
        else:
            self.model.eval()
            pred_list = None
            answer_list = None
            for i, batch in rec_data_iter:
                # 0. batch_data will be sent into the device(GPU or cpu)
                batch = tuple(t.to(self.device) for t in batch)
                user_ids, sasrec_input_ids, rec_answer, seq_position, target_neg, gru_input_ids, seq_len = batch
                rating_pred = self.model.predict(sasrec_input_ids, seq_position, gru_input_ids, seq_len)
                rating_pred = rating_pred.cpu().data.numpy().copy()
                batch_user_index = user_ids.cpu().numpy()
                rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0
                ind = np.argpartition(rating_pred, -20)[:, -20:]
                arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]
                if i == 0:
                    pred_list = batch_pred_list
                    answer_list = rec_answer.cpu().data
                else:
                    pred_list = np.append(pred_list, batch_pred_list, axis=0)
                    answer_list = np.append(answer_list, rec_answer.cpu().data, axis=0)
            answer_list = answer_list.tolist()
            answer_list = [[answer] for answer in answer_list]
            pred_list = pred_list.tolist()
            return self.get_full_sort_score(epoch, answer_list, pred_list)