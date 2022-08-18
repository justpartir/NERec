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
from param_parser import parameter_parser
from utils import tab_printer
from attentionwalk import AttentionWalkTrainer

class NERec(nn.Module):
    
    def __init__(self, node_enc, r2e):
        super(NERec, self).__init__()    
        self.node_enc = node_enc
        self.embed_dim = node_enc.embed_dim
        self.criterion = nn.MSELoss()

    def forward(self, nodes_u, nodes_v):
        
        embeds_u = self.node_enc(nodes_u, nodes_v, uv = False)
        embeds_v = self.node_enc(nodes_v, nodes_u, uv = True)
        
        scores = torch.mm(embeds_u, embeds_v.t()).diagonal()
        return scores


    def loss(self, nodes_u, nodes_v, labels_list):
        scores = self.forward(nodes_u, nodes_v)
        return self.criterion(scores, labels_list)
    
def train(model, device, train_loader, optimizer, epoch, best_rmse, best_mae):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        batch_nodes_u, batch_nodes_v, labels_list = data
        optimizer.zero_grad()
        loss = model.loss(batch_nodes_u.to(device), batch_nodes_v.to(device), labels_list.to(device))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 0:
            print('[%d, %5d] loss: %.3f, The best rmse/mae: %.6f / %.6f' % (
                epoch, i, running_loss / 100, best_rmse, best_mae))
            running_loss = 0.0
    return 0

def test(model, device, test_loader):
    model.eval()
    tmp_pred = []
    target = []
    with torch.no_grad():
        for test_u, test_v, tmp_target in test_loader:
            test_u, test_v, tmp_target = test_u.to(device), test_v.to(device), tmp_target.to(device)
            val_output = model.forward(test_u, test_v)
            tmp_pred.append(list(val_output.data.cpu().numpy()))
            target.append(list(tmp_target.data.cpu().numpy()))
    tmp_pred = np.array(sum(tmp_pred, []))
    target = np.array(sum(target, []))
    expected_rmse = sqrt(mean_squared_error(tmp_pred, target))
    mae = mean_absolute_error(tmp_pred, target)
    return expected_rmse, mae


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
    
    print('start')
    #args = parser.parse_known_args()[0]
    args = parameter_parser()
    tab_printer(args)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    
    dir_data = 'data/' + args.data
    path_data = dir_data + ".pkl"
    data_file = open(path_data,'rb')
    
    history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, traindata, validdata, testdata, social_adj_lists, item_adj_lists, ratings_list = pickle.load(
        data_file)

    traindata = np.array(traindata)
    validdata = np.array(validdata)
    testdata = np.array(testdata)   

    train_u = traindata[:, 0]
    train_v = traindata[:, 1]
    train_r = traindata[:, 2]
    valid_u = validdata[:, 0]
    valid_v = validdata[:, 1]
    valid_r = validdata[:, 2]
    test_u = testdata[:, 0]
    test_v = testdata[:, 1]
    test_r = testdata[:, 2]
    trainset = torch.utils.data.TensorDataset(torch.LongTensor(train_u), torch.LongTensor(train_v),
                                              torch.FloatTensor(train_r))
    validset = torch.utils.data.TensorDataset(torch.LongTensor(valid_u), torch.LongTensor(valid_v),
                                              torch.FloatTensor(valid_r))
    testset = torch.utils.data.TensorDataset(torch.LongTensor(test_u), torch.LongTensor(test_v),
                                             torch.FloatTensor(test_r))
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(validset, batch_size=args.test_batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True)

    num_users = history_u_lists.__len__()
    num_items = history_v_lists.__len__()
    num_ratings = ratings_list.__len__()
    
    

    embed_dim = args.embed_dim
    u2e = nn.Embedding(num_users, embed_dim).to(device)
    v2e = nn.Embedding(num_items, embed_dim).to(device)

    r2e = nn.Embedding(num_ratings + 1, embed_dim).to(device)
    

    model = AttentionWalkTrainer(args)
    model.fit()
    new_u2e = model.get_embedding()
    
    node_agg = UV_Aggregator(v2e, r2e, u2e, new_u2e, embed_dim, r2e.num_embeddings - 1, cuda=device)

    node_enc = UV_Encoder(u2e, v2e, embed_dim, history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, social_adj_lists, item_adj_lists, node_agg, percent=args.percent,  cuda=device)

    #NERec model
    nerec = NERec(node_enc, r2e).to(device)
    
    #optimizer:Adam,衰减率初始为0.0001
    optimizer = torch.optim.Adam(nerec.parameters(), lr=args.lr, weight_decay = args.weight_decay)
    #optimizer = Ranger(nerec.parameters(), lr=args.lr, weight_decay = args.weight_decay)
    

    if args.load_from_checkpoint == True:
        checkpoint = torch.load('models/' + args.data + '.pt')
        nerec.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])    
        
    best_rmse = 9999.0
    best_mae = 9999.0

    endure_count = 0

    for epoch in range(1, args.epochs + 1):

        train(nerec, device, train_loader, optimizer, epoch, best_rmse, best_mae)
        expected_rmse, mae = test(nerec, device, valid_loader)
        if best_rmse > expected_rmse:
            best_rmse = expected_rmse
            best_mae = mae
            endure_count = 0
            torch.save({
            'epoch': epoch,
            'model_state_dict': nerec.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, 'models/' + args.data + '.pt')
        else:
            endure_count += 1
        print("NERec rmse on valid set: %.4f, mae:%.4f " % (expected_rmse, mae))
        rmse, mae = test(nerec, device, test_loader)
        print('NERec rmse on test set: %.4f, mae:%.4f '%(rmse, mae))

        if endure_count > 5:
            break
    print('finished')

if __name__ == "__main__":
    main()