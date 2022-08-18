import torch
import torch.nn as nn
import torch.nn.functional as F

class UV_Aggregator(nn.Module):
    """
    item and user aggregator: for aggregating embeddings of neighbors (item/user aggreagator).
    """

    def __init__(self, v2e, r2e, u2e, new_u2e, embed_dim, relation_token, cuda="cuda"):
        super(UV_Aggregator, self).__init__()
        self.v2e = v2e
        self.r2e = r2e
        self.u2e = u2e
        self.new_u2e = new_u2e
        self.relation_token = relation_token
        self.device = cuda
        self.embed_dim = embed_dim
        self.relation_att = nn.Parameter(torch.randn(2 * embed_dim, requires_grad=True).to(self.device))
        self.linear = nn.Linear(2 * embed_dim, embed_dim)
        self.softmax1 = nn.Softmax(dim = 0)
        self.softmax2 = nn.Softmax(dim = 0)

    def neighbor_agg(self, query, history_feature, relation_feature, percent):
        #item与user，user与item
        
        #查询策略
        
        #Neighbor-Difference method
        prob = -torch.norm(query - history_feature, dim = 1)
        
        #Neighbor-Intersection method
        #prob2 = F.cosine_similarity(query, history_feature, dim=-1)
 
        
        prob = self.softmax1(prob)
        neighbor_selected = torch.multinomial(prob, max(1,int(percent * len(history_feature))))
        relation_selected = relation_feature[neighbor_selected]
        neighbor_selected = history_feature[neighbor_selected]
        selected = torch.cat((neighbor_selected, relation_selected), 1)
        selected = torch.mm(selected, self.relation_att.unsqueeze(0).t()).squeeze(-1)
        prob = self.softmax2(selected)
        return torch.mm(neighbor_selected.transpose(0,1), prob.unsqueeze(-1)).squeeze(-1)
      
    def forward(self, self_feats, target_feats, history_uv, history_r, adj, uv, percent):

        embed_matrix = torch.zeros(len(history_uv), self.embed_dim, dtype=torch.float).to(self.device)

        #Query Layer
        #新的query层 新的user的embedding,对user类型的节点需要聚合历史交互的item,在Encoders中实现
        #print(self_feats)
        #print(target_feats)
        #if uv is True: #item类型节点          
            #query = self.linear(torch.cat((self_feats, target_feats), dim = -1))
        #else:#user类型节点
            #query = self.linear(torch.cat((self_feats, target_feats), dim = -1))
        query = self.linear(torch.cat((self_feats, target_feats), dim = -1))
        
        #Neighbor Sampling and Aggregation
        for i in range(len(history_uv)):
            #比如item，history代表一个item的user邻居集合
            history = history_uv[i]
            #邻居集合长度、relation长度
            num_histroy_item = len(history)
            #relation集合
            tmp_label = history_r[i]

            if uv is True:#item类型节点
                e_uv = self.u2e.weight[history]#item的user邻居
                e_neighbor = self.v2e.weight[adj[i]]#item的item邻居
                #两种邻居分开进行不同的处理
                e_uv = torch.cat((e_uv, e_neighbor), 0)
                tmp_label += [self.relation_token] * len(adj[i])
                num_histroy_item += len(adj[i])

            else:
                e_uv = self.v2e.weight[history]#user的item邻居
                #e_neighbor = self.u2e.weight[adj[i]]#user的user邻居
                newu = torch.from_numpy(self.new_u2e).to(self.device)
                e_neighbor = newu[adj[i]]#user的user邻居
                #两种邻居分开进行不同的处理
                e_uv = torch.cat((e_uv, e_neighbor), 0)
                tmp_label += [self.relation_token] * len(adj[i])
                num_histroy_item += len(adj[i])

            e_r = self.r2e.weight[tmp_label]
            if num_histroy_item != 0:
                #agg = self.neighbor_agg2(query[i], e_uv, e_neighbor, e_r, percent)
                #embed_matrix[i] = agg
                if uv is False:#user的两个集合的处理
                    #agg_u_i = self.neighbor_agg(query[i], e_uv, e_r, percent)
                    #agg_u_u = self.neighbor_agg(query[i], e_neighbor, e_r, percent)
                    #agg_u_u = self.neighbor_agg_u_u(query[i], e_neighbor, percent)
                    #对于user和user来说，此处需要变
                    #e_uv = torch.cat((e_uv, e_neighbor), 0)
                    agg_u = self.neighbor_agg(query[i], e_uv, e_r, percent)
                    embed_matrix[i] = agg_u
                    #embed_matrix[i] = torch.cat((agg_u_i, agg_u_u), 0)
                else:#item的两个集合的处理
                    #e_uv = torch.cat((e_uv, e_neighbor), 0)
                    agg_i = self.neighbor_agg(query[i], e_uv, e_r, percent)
                    embed_matrix[i] = agg_i

        to_feats = embed_matrix
        return to_feats
