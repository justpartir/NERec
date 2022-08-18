import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import sys
import pickle


class UV_Encoder(nn.Module):

    def __init__(self, u2e, v2e, embed_dim, history_u_lists, history_ur_lists,
                history_v_lists, history_vr_lists, social_adj_lists, item_adj_lists, aggregator, percent, cuda="cuda"):
        super(UV_Encoder, self).__init__()

        self.u2e = u2e
        self.v2e = v2e
        self.history_u_lists = history_u_lists
        self.history_ur_lists = history_ur_lists
        self.history_v_lists = history_v_lists
        self.history_vr_lists = history_vr_lists
        self.social_adj_lists = social_adj_lists
        self.item_adj_lists = item_adj_lists
        
        self.aggregator = aggregator
        self.embed_dim = embed_dim
        self.device = cuda
        
        self.p = percent

        
        #concat后对特征进行操作
        self.linear1 = nn.Linear(2 * self.embed_dim, self.embed_dim)     
        #self.linear2 = nn.Linear(2 * self.embed_dim, self.embed_dim)
        #self.bn1 = nn.BatchNorm1d(self.embed_dim)
        #self.bn2 = nn.BatchNorm1d(self.embed_dim)

        

    def forward(self, nodes, nodes_target, uv):
        tmp_history_uv = []
        tmp_history_r = []
        tmp_adj = []
        for i in range(len(nodes)):
            if uv == True:#item
                #有交互的user
                tmp_history_uv.append(self.history_v_lists[int(nodes[i])])
                #对应的relation
                tmp_history_r.append(self.history_vr_lists[int(nodes[i])])
                #item邻居
                tmp_adj.append(list(self.item_adj_lists[int(nodes[i])]))
                self_feats = self.v2e.weight[nodes]
                target_feats = self.u2e.weight[nodes_target]
            else :#user
                #有交互的item
                tmp_history_uv.append(self.history_u_lists[int(nodes[i])])
                #print(tmp_history_uv)
                #对应的relation
                tmp_history_r.append(self.history_ur_lists[int(nodes[i])])
                #user邻居
                tmp_adj.append(list(self.social_adj_lists[int(nodes[i])]))
                #构造查询层，对user类型的节点需要聚合历史交互的item
                item_neighbor = self.v2e.weight[self.history_u_lists[int(nodes[i])]]
                self_feats = self.u2e.weight[nodes]
                #self_feats = self.neighbor_agg_hisact(self_feats, item_neighbor)
                #self_feats = torch.cat((self_feats, item_neighbor), 1)
                #self_feats = F.relu(self.linear1(self_feats))                
                target_feats = self.v2e.weight[nodes_target]
                self_feats = torch.cat((self_feats, target_feats), 1)
                self_feats = F.relu(self.linear1(self_feats))
                
        neigh_feats = self.aggregator.forward(self_feats, target_feats, tmp_history_uv, tmp_history_r, tmp_adj, uv, self.p)
        
        #不需要改动，concat当前自己的embedding和聚合得到的邻居的embedding
        combined = torch.cat((self_feats, neigh_feats), dim = -1)
        combined = F.relu(self.linear1(combined))


        return combined
