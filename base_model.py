import torch
import torch.nn as nn
import torch.nn.functional as nnf
from config import config
from torch.autograd import Variable
from torch.optim import Adam
from metrics import mrr_mr_hitk
from data_utils import batch_by_size
import logging
import os


class BaseModule(nn.Module):
    def __init__(self):
        super(BaseModule, self).__init__()

    def score(self, src, rel, dst, tem):
        raise NotImplementedError

    def dist(self, src, rel, dst, tem):
        raise NotImplementedError

    def prob_logit(self, src, rel, dst, tem):
        raise NotImplementedError

    def prob(self, src, rel, dst, tem):
        return nnf.softmax(self.prob_logit(src, rel, dst, tem))

    def constraint(self):
        pass

    def pair_loss(self, src, rel, dst, src_bad, dst_bad, tem):
        d_good = self.dist(src, rel, dst, tem)
        d_bad = self.dist(src_bad, rel, dst_bad, tem)
        return nnf.relu(self.margin + d_good - d_bad)

    def softmax_loss(self, src, rel, dst, tem, truth):
        probs = self.prob(src, rel, dst, tem)
        n = probs.size(0)
        truth_probs = torch.log(probs[torch.arange(0, n).type(torch.LongTensor).cuda(), truth] + 1e-30)
        return -truth_probs

    def get_rseq(self, pos_r, pos_tem):
        # print(pos_r.size()) [728, 20]
        # print(pos_tem.size()) [728, 20, 7]
        pos_r_e = self.rel_embed(pos_r)
        pos_r_e = pos_r_e.unsqueeze(0).transpose(0, 1)
        pos_r_e = pos_r_e.transpose(1, 2)

        bs = pos_tem.size(0)  # batch size
        n_sample = pos_tem.size(1)
        pos_tem = pos_tem.contiguous()
        pos_tem = pos_tem.view(bs * n_sample, -1)
        token_e = self.tem_embed(pos_tem)
        token_e = token_e.view(bs, n_sample, -1, self.config.dim)
        pos_seq_e = torch.cat((pos_r_e, token_e), 2)
        pos_seq_e = pos_seq_e.view(bs * n_sample, -1, self.config.dim)
        # print(pos_seq_e.size())

        hidden_tem = self.lstm(pos_seq_e)
        hidden_tem = hidden_tem[0, :, :]
        # pos_rseq_e = hidden_tem
        pos_rseq_e = hidden_tem.view(bs, n_sample, -1, self.config.dim)

        # print(pos_rseq_e)
        return pos_rseq_e


class BaseModel(object):
    def __init__(self):
        self.mdl = None # type: BaseModule
        self.weight_decay = 0

    def save(self, filename):
        torch.save(self.mdl.state_dict(), filename)

    def load(self, filename):
        self.mdl.load_state_dict(torch.load(filename, map_location=lambda storage, location: storage.cuda()))

    def gen_step(self, src, rel, dst, n_sample=1, temperature=1.0, train=True):
        if not hasattr(self, 'opt'):
            self.opt = Adam(self.mdl.parameters(), weight_decay=self.weight_decay)
        n, m = dst.size()
        rel_var = Variable(rel.cuda())
        src_var = Variable(src.cuda())
        dst_var = Variable(dst.cuda())

        logits = self.mdl.prob_logit(src_var, rel_var, dst_var) / temperature
        probs = nnf.softmax(logits)
        row_idx = torch.arange(0, n).type(torch.LongTensor).unsqueeze(1).expand(n, n_sample)
        sample_idx = torch.multinomial(probs, n_sample, replacement=True)
        sample_srcs = src[row_idx, sample_idx.data.cpu()]
        sample_dsts = dst[row_idx, sample_idx.data.cpu()]
        rewards = yield sample_srcs, sample_dsts
        if train:
            self.mdl.zero_grad()
            log_probs = nnf.log_softmax(logits)
            reinforce_loss = -torch.sum(Variable(rewards) * log_probs[row_idx.cuda(), sample_idx.data])
            reinforce_loss.backward()
            self.opt.step()
            self.mdl.constraint()
        yield None

    def dis_step(self, src, rel, dst, src_fake, dst_fake, tem, train=True):
        if not hasattr(self, 'opt'):
            self.opt = Adam(self.mdl.parameters(), weight_decay=self.weight_decay)
        src_var = Variable(src.cuda())
        rel_var = Variable(rel.cuda())
        dst_var = Variable(dst.cuda())
        src_fake_var = Variable(src_fake.cuda())
        dst_fake_var = Variable(dst_fake.cuda())
        tem_var = Variable(tem.cuda())
        losses = self.mdl.pair_loss(src_var, rel_var, dst_var, src_fake_var, dst_fake_var, tem_var)
        fake_scores = self.mdl.score(src_fake_var, rel_var, dst_fake_var, tem_var)
        if train:
            self.mdl.zero_grad()
            torch.sum(losses).backward()
            self.opt.step()
            self.mdl.constraint()
        return losses.data, -fake_scores.data

    def test_link(self, test_data, n_ent, heads, tails, filt=True):
        mrr_tot = 0
        mr_tot = 0
        hit1_tot = 0
        hit3_tot = 0
        hit10_tot = 0
        count = 0
        self.mdl.eval()
        for batch_s, batch_r, batch_t, batch_tem in batch_by_size(config().test_batch_size, *test_data):
            batch_size = batch_s.size(0)
            rel_var = Variable(batch_r.unsqueeze(1).expand(batch_size, n_ent).cuda(), volatile=True)
            src_var = Variable(batch_s.unsqueeze(1).expand(batch_size, n_ent).cuda(), volatile=True)
            dst_var = Variable(batch_t.unsqueeze(1).expand(batch_size, n_ent).cuda(), volatile=True)
            tem_last_dim = batch_tem.size(-1)
            tem_var = Variable(batch_tem.unsqueeze(1).expand(batch_size, n_ent, tem_last_dim).cuda(), volatile=True)
            all_var = Variable(torch.arange(0, n_ent).unsqueeze(0).expand(batch_size, n_ent)
                               .type(torch.LongTensor).cuda(), volatile=True)
            self.mdl.zero_grad()
            batch_dst_scores = self.mdl.score(src_var, rel_var, all_var, tem_var).data
            batch_src_scores = self.mdl.score(all_var, rel_var, dst_var, tem_var).data
            for s, r, t, dst_scores, src_scores in zip(batch_s, batch_r, batch_t, batch_dst_scores, batch_src_scores):
                if filt:
                    if tails[(s, r)]._nnz() > 1:
                        tmp = dst_scores[t]
                        dst_scores += tails[(s, r)].cuda() * 1e30
                        dst_scores[t] = tmp
                    if heads[(t, r)]._nnz() > 1:
                        tmp = src_scores[s]
                        src_scores += heads[(t, r)].cuda() * 1e30
                        src_scores[s] = tmp
                mrr, mr, hit1, hit3, hit10 = mrr_mr_hitk(dst_scores, t)
                mrr_tot += mrr
                mr_tot += mr
                hit1_tot += hit1
                hit3_tot += hit3
                hit10_tot += hit10
                mrr, mr, hit1, hit3, hit10 = mrr_mr_hitk(src_scores, s)
                mrr_tot += mrr
                mr_tot += mr
                hit1_tot += hit1
                hit3_tot += hit3
                hit10_tot += hit10
                count += 2
        logging.info('Test_MRR=%f, Test_MR=%f, Test_H@1=%f, Test_H@3=%f, Test_H@10=%f', mrr_tot / count, mr_tot / count, hit1_tot / count, hit3_tot / count, hit10_tot / count)
        writeList = ['testSet', '%.6f' % (hit1_tot / count), '%.6f' % (hit3_tot / count), '%.6f' % (hit10_tot / count), '%.6f' % (mr_tot / count),
                     '%.6f' % (mrr_tot / count)]
        # Write the result into file
        with open(os.path.join('./result/', config().task.dir.split('/')[-1] + '_' + config().pretrain_config), 'a') as fw:
            fw.write('\t'.join(writeList) + '\n')
        return mrr_tot / count
