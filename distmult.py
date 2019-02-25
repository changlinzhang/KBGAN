import torch as t
import torch.nn as nn
import torch.nn.functional as f
from config import config
from torch.optim import Adam, SGD, Adagrad
from torch.autograd import Variable
from data_utils import batch_by_num
from base_model import BaseModel, BaseModule
from LSTMLinear import LSTMModel
import logging
import os

class DistMultModule(BaseModule):
    def __init__(self, n_ent, n_rel, config):
        super(DistMultModule, self).__init__()
        sigma = 0.2
        n_tem = 32 # can be made to config for future datasets
        self.config = config
        self.rel_embed = nn.Embedding(n_rel, config.dim)
        self.rel_embed.weight.data.div_((config.dim / sigma ** 2) ** (1 / 6))
        self.ent_embed = nn.Embedding(n_ent, config.dim)
        self.ent_embed.weight.data.div_((config.dim / sigma ** 2) ** (1 / 6))
        self.tem_embed = nn.Embedding(n_tem, config.dim)
        self.tem_embed.weight.data.div_((config.dim / sigma ** 2) ** (1 / 6))
        self.lstm = LSTMModel(config.dim, n_layer=1)

    def forward(self, src, rel, dst, tem):
        rseq_e = self.get_rseq(rel, tem)
        tmp = self.ent_embed(dst) * self.ent_embed(src)
        return t.sum(tmp * rseq_e, dim=-1)

    def score(self, src, rel, dst, tem):
        return -self.forward(src, rel, dst, tem)

    def dist(self, src, rel, dst, tem):
        return -self.forward(src, rel, dst, tem)

    def prob_logit(self, src, rel, dst, tem):
        return self.forward(src, rel, dst, tem)

class DistMult(BaseModel):
    def __init__(self, n_ent, n_rel, config):
        super(DistMult, self).__init__()
        self.mdl = DistMultModule(n_ent, n_rel, config)
        self.mdl.cuda()
        self.config = config
        self.weight_decay = config.lam / config.n_batch

    def pretrain(self, train_data, corrupter, tester=None):
        src, rel, dst, tem = train_data
        n_train = len(src)
        n_epoch = self.config.n_epoch
        n_batch = self.config.n_batch
        optimizer = Adam(self.mdl.parameters(), weight_decay=self.weight_decay)
        best_perf = 0
        for epoch in range(n_epoch):
            epoch_loss = 0
            if epoch % self.config.sample_freq == 0:
                rand_idx = t.randperm(n_train)
                src = src[rand_idx]
                rel = rel[rand_idx]
                dst = dst[rand_idx]
                tem = tem[rand_idx]
                src_corrupted, rel_corrupted, dst_corrupted, tem_corrupted = corrupter.corrupt(src, rel, dst, tem)
                src_corrupted = src_corrupted.cuda()
                rel_corrupted = rel_corrupted.cuda()
                dst_corrupted = dst_corrupted.cuda()
                tem_corrupted = tem_corrupted.cuda()
            for ss, rs, ts, tems in batch_by_num(n_batch, src_corrupted, rel_corrupted, dst_corrupted, tem_corrupted, n_sample=n_train):
                self.mdl.zero_grad()
                label = t.zeros(len(ss)).type(t.LongTensor).cuda()
                loss = t.sum(self.mdl.softmax_loss(Variable(ss), Variable(rs), Variable(ts), Variable(tems), label))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.data[0]
            logging.info('Epoch %d/%d, Loss=%f', epoch + 1, n_epoch, epoch_loss / n_train)
            if (epoch + 1) % self.config.epoch_per_test == 0:
                # test_perf = tester()
                # if test_perf > best_perf:
                self.save(os.path.join(config().task.dir, self.config.model_file))
                    # best_perf = test_perf
        return best_perf
