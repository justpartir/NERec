#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
import

导入所需要的所有包

'''
import os
import sys
import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import pickle
import numpy as np
import pandas as pd
import time
import random
from collections import defaultdict
from UV_Encoders import UV_Encoder
from UV_Aggregators import UV_Aggregator

import torch.nn.functional as F
import torch.utils.data
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
import datetime
import argparse

from texttable import Texttable
from param_parsers import parameter_parsers
from utils import tab_printer
from attentionwalk import AttentionWalkTrainer

from metrics import MetronAtK


# In[2]:


'''
NERec模型主体
ranking task和rating task的模型主体不同
ranking的比较目标是label，计算出预测分数后还需要sigmoid归一化

'''

class NERec(nn.Module):
    
    def __init__(self, node_enc, r2e):
        super(NERec, self).__init__()    
        #和GraphRec不同，由于部分item相连，使得item也有两种邻居，将user和item节点视作一类，后面在模型里对user和item不同的处理则根据uv这个变量进行分类
        self.node_enc = node_enc
        self.embed_dim = node_enc.embed_dim
        #MLP多层感知器
        #self.w_ur1 = nn.Linear(self.embed_dim, self.embed_dim)
        #self.w_ur2 = nn.Linear(self.embed_dim, self.embed_dim)
        #self.w_vr1 = nn.Linear(self.embed_dim, self.embed_dim)
        #self.w_vr2 = nn.Linear(self.embed_dim, self.embed_dim)
        #self.w_uv1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        #self.w_uv2 = nn.Linear(self.embed_dim, 16)
        #self.w_uv3 = nn.Linear(16, 1)
        #self.r2e = r2e
        #self.bn1 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        #self.bn2 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        #self.bn3 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        #self.bn4 = nn.BatchNorm1d(16, momentum=0.5)      
        #loss函数使用torch.nn自带的损失函数，详细介绍见https://pytorch-cn.readthedocs.io/zh/latest/package_references/torch-nn/
        #self.criterion = nn.MSELoss()
        self.criterion = nn.BCELoss()

    def forward(self, nodes_u, nodes_v):
        
        #利用uv变量区分user和item节点，False为user节点，True为item节点
        embeds_u = self.node_enc(nodes_u, nodes_v, uv = False)
        embeds_v = self.node_enc(nodes_v, nodes_u, uv = True)
        
        #向量点积
        scores = torch.mm(embeds_u, embeds_v.t()).diagonal()
        scores = torch.sigmoid(scores)
        return scores

        #MLP多层感知器
        #x_u = F.relu(self.bn1(self.w_ur1(embeds_u)))
        #x_u = F.dropout(x_u, training=self.training)
        #x_u = self.w_ur2(x_u)
        #x_v = F.relu(self.bn2(self.w_vr1(embeds_v)))
        #x_v = F.dropout(x_v, training=self.training)
        #x_v = self.w_vr2(x_v)

        #x_uv = torch.cat((x_u, x_v), 1)
        #x = F.relu(self.bn3(self.w_uv1(x_uv)))
        #x = F.dropout(x, training=self.training)
        #x = F.relu(self.bn4(self.w_uv2(x)))
        #x = F.dropout(x, training=self.training)
        #scores = self.w_uv3(x)

    def loss(self, nodes_u, nodes_v, labels_list):
        scores = self.forward(nodes_u, nodes_v)
        return self.criterion(scores.view(-1), labels_list)


# In[3]:


'''
训练模型以及对验证集和测试集测试计算指标

optimizer选择：Adam，学习率初始为0.001，衰减率初始为00001
使用方法：
torch.optim.Adam(NERec.parameters, lr, weight_decay)
torch.optim中文文档：
https://pytorch-cn.readthedocs.io/zh/latest/package_references/torch-optim/

'''

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        batch_nodes_u, batch_nodes_v, ratings_list, labels_list = data
        optimizer.zero_grad()
        loss = model.loss(batch_nodes_u.to(device), batch_nodes_v.to(device), labels_list.to(device))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 0:
            print('[%d, %5d] loss: %.3f' % (
                epoch, i, running_loss / 100))
            running_loss = 0.0
    return 0

def test(model, device, test_loader, _metron):
    model.eval()
    tmp_pred = []
    target = []
    user, item, rating, label, c_label = [], [], [], [], []
    with torch.no_grad():
        for test_u, test_v, tmp_target, tmp_label in test_loader:
            #tmp_label就是真实的label
            test_u, test_v, tmp_target, tmp_label = test_u.to(device), test_v.to(device), tmp_target.to(device), tmp_label.to(device)
            #预测的label，介于0-1之间
            val_output = model.forward(test_u, test_v)
            
            for i in test_u.tolist():
                user.append(i)
            for i in test_v.tolist():
                item.append(i)
            for i in tmp_target.tolist():
                rating.append(i)
            for i in tmp_label.tolist():
                label.append(i)
            for i in val_output.data.view(-1).tolist():
                c_label.append(i)
        
        _metron.subjects = [user, item, rating, c_label, label]
    recall, ndcg = _metron.cal_recall(), _metron.cal_ndcg()
    return recall, ndcg
            #tmp_pred.append(list(val_output.data.cpu().numpy()))
            #真实的label，0或者1
            #label.append(list(tmp_label.data.cpu().numpy()))
    #tmp_pred = np.array(sum(tmp_pred, []))
    #label = np.array(sum(label, []))
    #ranking task不同于rating task，评价指标为NDCG命中率，Recall召回率
    #expected_rmse = sqrt(mean_squared_error(tmp_pred, target))
    #mae = mean_absolute_error(tmp_pred, target)
    #return expected_rmse, mae


# In[4]:


'''
main函数，模型具体的过程


'''
def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]])
    t.add_rows([[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())

def main():
    
    #args = parser.parse_known_args()[0]
    args = parameter_parsers()
    tab_printer(args)
    
    #使用GPU训练，要找机器的GPU序号
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    #输出一下当前是否使用到了GPU训练模型
    print(device)
    
    #读取数据文件
    dir_data = 'data/' + args.data
    path_data = dir_data + ".pkl"
    data_file = open(path_data,'rb')
    
    #读取数据
    history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, traindata, validdata, testdata, social_adj_lists, item_adj_lists, ratings_list = pickle.load(
        data_file)
    #np.array()产生数组对象
    traindata = np.array(traindata)
    validdata = np.array(validdata)
    testdata = np.array(testdata)   
    #X[:,0]取二维数组中的第一列所有行的数据，以此类推
    train_u = traindata[:, 0]
    train_v = traindata[:, 1]
    train_r = traindata[:, 2]
    train_label = traindata[:, 3]
    
    valid_u = validdata[:, 0]
    valid_v = validdata[:, 1]
    valid_r = validdata[:, 2]
    valid_label = validdata[:, 3]
    
    test_u = testdata[:, 0]
    test_v = testdata[:, 1]
    test_r = testdata[:, 2]
    test_label = testdata[:, 3]
    
    trainset = torch.utils.data.TensorDataset(torch.LongTensor(train_u), torch.LongTensor(train_v),
                                              torch.FloatTensor(train_r), torch.FloatTensor(train_label))
    validset = torch.utils.data.TensorDataset(torch.LongTensor(valid_u), torch.LongTensor(valid_v),
                                              torch.FloatTensor(valid_r), torch.FloatTensor(valid_label))
    testset = torch.utils.data.TensorDataset(torch.LongTensor(test_u), torch.LongTensor(test_v),
                                             torch.FloatTensor(test_r), torch.FloatTensor(test_label))
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(validset, batch_size=args.test_batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True)
    #user,item和rating的数量
    num_users = history_u_lists.__len__()
    num_items = history_v_lists.__len__()
    num_ratings = ratings_list.__len__()
    
    
    #嵌入层
    embed_dim = args.embed_dim
    u2e = nn.Embedding(num_users, embed_dim).to(device)
    v2e = nn.Embedding(num_items, embed_dim).to(device)
    #注意在对评分进行嵌入时，需要考虑0评分，即在rating_list中没有的评分，所以要num_ratings + 1
    r2e = nn.Embedding(num_ratings + 1, embed_dim).to(device)
    
    #辅助的社交图模块，生成的是更新的u2e.weight
    model = AttentionWalkTrainer(args)
    model.fit()
    new_u2e = model.get_embedding()
    
    
    #两个embedding的比较
    #print(u2e.weight[0])
    #print(new_u2e[0])
    #print(len(u2e.weight))
    #print(len(new_u2e))
    
   
    node_agg = UV_Aggregator(v2e, r2e, u2e, new_u2e, embed_dim, r2e.num_embeddings - 1, cuda=device)
    #print(1)
    node_enc = UV_Encoder(u2e, v2e, embed_dim, history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, social_adj_lists, item_adj_lists, node_agg, percent=args.percent,  cuda=device)
    #print(2)
    #NERec model
    nerec = NERec(node_enc, r2e).to(device)
    
    #optimizer:Adam,衰减率初始为0.0001
    optimizer = torch.optim.Adam(nerec.parameters(), lr=args.lr, weight_decay = args.weight_decay)
    #optimizer = Ranger(nerec.parameters(), lr=args.lr, weight_decay = args.weight_decay)
    
    #如果有程序断点，并且--load_from_checkpoint设置为True，则读取断点时保存的模型并加载
    if args.load_from_checkpoint == True:
        checkpoint = torch.load('models/' + args.data + '.pt')
        nerec.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])    
        
    best_rmse = 9999.0
    best_mae = 9999.0
    #设置一个训练次数的参数，如果训练的目标5轮没有提升，自动停止程序
    endure_count = 0

    #把新的评价指标的评测函数放到一个_metron中，用外部的类来进行运行
    _metron = MetronAtK(top_k=5)
    
    index_sum = []
    pre_sum = 0
    best_sum = 0
    for epoch in range(1, args.mepochs + 1):

        train(nerec, device, train_loader, optimizer, epoch)
        #ranking的评价指标和rating不同，不是rmse 和MAE
        #先验证，再test
        expected_recall, expected_ndcg = test(nerec, device, valid_loader, _metron)
        if epoch == 0:
            pre_sum = expected_recall + expected_ndcg
            index_sum.append(0)
        else:
            if expected_recall + expected_ndcg < pre_sum:
                index_sum.append(1)
            else:
                pre_sum = expected_recall + expected_ndcg
                index_sum.append(0)
        if sum(index_sum[-10:]) == 10:
            break
        if epoch == 0:
            best_sum = expected_recall + expected_ndcg
            torch.save({
            'epoch': epoch,
            'model_state_dict': nerec.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, 'models/' + args.data + '.pt')
        elif expected_recall + expected_ndcg > best_sum:
            best_sum = expected_recall + expected_ndcg
            torch.save({
            'epoch': epoch,
            'model_state_dict': nerec.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, 'models/' + args.data + '.pt')
        #if best_recall > expected_recall:
            #best_recall = expected_recall
            #best_mae = mae
            #endure_count = 0
            #torch.save({
            #'epoch': epoch,
            #'model_state_dict': nerec.state_dict(),
            #'optimizer_state_dict': optimizer.state_dict(),
            #}, 'models/' + args.data + '.pt')
        #else:
            #endure_count += 1
        print("NERec Recall on valid set: %.4f, NDCG:%.4f " % (expected_recall, expected_ndcg))
        t_recall, t_ndcg = test(nerec, device, test_loader, _metron)
        print('NERec Recall on test set: %.4f, NDCG:%.4f '%(t_recall, t_ndcg))

    print('finished')

if __name__ == "__main__":
    main()


# In[ ]:


import torch
torch.cuda.is_available()

