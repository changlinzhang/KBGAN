import os
import logging
import torch
from corrupter import BernCorrupter, BernCorrupterMulti
from read_data import index_ent_rel, graph_size, read_data
from config import config, overwrite_config_with_args
from logger_init import logger_init
from data_utils import inplace_shuffle, heads_tails
from select_gpu import select_gpu
from trans_e import TransE
from trans_d import TransD
from distmult import DistMult
from tadistmult import TADistMult
from compl_ex import ComplEx

logger_init()
torch.cuda.set_device(select_gpu())
overwrite_config_with_args()

task_dir = config().task.dir
# kb_index = index_ent_rel(os.path.join(task_dir, 'train2id.txt'),
#                          os.path.join(task_dir, 'valid2id.txt'),
#                          os.path.join(task_dir, 'test2id.txt'))
kb_index = index_ent_rel(os.path.join(task_dir, 'train2id.txt'),
                         os.path.join(task_dir, 'test2id.txt'))
n_ent, n_rel = graph_size(kb_index)
print(n_ent, n_rel)

train_data = read_data(os.path.join(task_dir, 'train2id.txt'), kb_index, os.path.join(task_dir, 'train_tem.npy'))
inplace_shuffle(*train_data)
# valid_data = read_data(os.path.join(task_dir, 'valid2id.txt'), kb_index)
test_data = read_data(os.path.join(task_dir, 'test2id.txt'), kb_index, os.path.join(task_dir, 'test_tem.npy'))
# heads, tails = heads_tails(n_ent, train_data, valid_data, test_data)
heads, tails = heads_tails(n_ent, train_data, test_data)
# valid_data = [torch.LongTensor(vec) for vec in valid_data]
test_data = [torch.LongTensor(vec) for vec in test_data]
# tester = lambda: gen.test_link(valid_data, n_ent, heads, tails)
train_data = [torch.LongTensor(vec) for vec in train_data]

mdl_type = config().pretrain_config
gen_config = config()[mdl_type]
if mdl_type == 'TransE':
    corrupter = BernCorrupter(train_data, n_ent, n_rel)
    gen = TransE(n_ent, n_rel, gen_config)
elif mdl_type == 'TransD':
    corrupter = BernCorrupter(train_data, n_ent, n_rel)
    gen = TransD(n_ent, n_rel, gen_config)
elif mdl_type == 'DistMult':
    corrupter = BernCorrupterMulti(train_data, n_ent, n_rel, gen_config.n_sample)
    gen = DistMult(n_ent, n_rel, gen_config)
elif mdl_type == 'TADistMult':
    corrupter = BernCorrupterMulti(train_data, n_ent, n_rel, gen_config.n_sample)
    gen = TADistMult(n_ent, n_rel, gen_config)
elif mdl_type == 'ComplEx':
    corrupter = BernCorrupterMulti(train_data, n_ent, n_rel, gen_config.n_sample)
    gen = ComplEx(n_ent, n_rel, gen_config)
# gen.pretrain(train_data, corrupter, tester)
model_path_name = os.path.join(task_dir, config().task.dir.split('/')[-1] + '_' + gen_config.model_file)
if config().train == 1:
    if os.path.exists(model_path_name):
        gen.load(model_path_name)
    gen.pretrain(train_data, corrupter)
gen.load(model_path_name)
gen.test_link(test_data, n_ent, heads, tails)
