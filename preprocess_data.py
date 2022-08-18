import scipy.io as scio
import numpy as np
from collections import defaultdict
import pickle

print('start')
path = 'data/ciao'
trust = scio.loadmat(path + '/trustnetwork.mat')
trust = trust['trustnetwork']

rating = scio.loadmat(path + '/rating.mat')
rating = rating['rating']

delete = []
for i in range(len(trust)):
    if trust[i,0] == trust[i,1]:
        delete.append(i)
trust = np.delete(trust,delete,0)

dic_user_o2i = {}
i = 0

for user in trust.reshape(-1,):
  if user in dic_user_o2i.keys():
    continue
  else:
    dic_user_o2i[user] = i
    i += 1


delete = []
for i in range(len(rating)):
  if rating[i,0] not in dic_user_o2i.keys():
    delete.append(i)
rating = np.delete(rating, delete, 0)


dic_item_o2i = {}
i = 0
for item in rating[:, 1]:
  if item in dic_item_o2i.keys():
    continue
  else:
    dic_item_o2i[item] = i
    i += 1


history_u_lists = defaultdict()
history_ur_lists = defaultdict()
history_v_lists = defaultdict()
history_vr_lists = defaultdict()
social_adj_lists = defaultdict()
item_adj_lists = defaultdict()
ratings_list = defaultdict()


    

data_total = []

for user in range(len(dic_user_o2i)):
  social_adj_lists[user] = []
for line in trust:
  social_adj_lists[dic_user_o2i[line[0]]].append(dic_user_o2i[line[1]])
  social_adj_lists[dic_user_o2i[line[1]]].append(dic_user_o2i[line[0]])

for user in range(len(dic_user_o2i)):
  social_adj_lists[user] = set(social_adj_lists[user])



i = 0
for rate in set(rating[:, 3]):
  ratings_list[rate] = i
  i += 1
  

for user in range(len(dic_user_o2i)):
  history_u_lists[user] = []
  history_ur_lists[user] = []
for item in range(len(dic_item_o2i)):
  history_v_lists[item] = []
  history_vr_lists[item] = []
  
rating = rating[:, [0, 1, 3]]
np.random.shuffle(rating)
valid = rating[ : int(0.2 * len(rating))]
test = rating[int(0.2 * len(rating)) : int(0.4 * len(rating))]
train = rating[int(0.4 * len(rating)): ]

def build_item_adj_lists(history_v_lists):
    adj_lists = {}
    for key in history_v_lists.keys():
        adj_lists[key] = []
    for key in history_v_lists.keys():
        for key_temp in history_v_lists.keys():
            if key != key_temp and len(set(history_v_lists[key]) | set(history_v_lists[key_temp])) != 0:
                if len(set(history_v_lists[key]) & set(history_v_lists[key_temp])) / len(set(history_v_lists[key]) | set(history_v_lists[key_temp])) > 0.5:
                    adj_lists[key].append(key_temp)
    return adj_lists

validset = []
testset = []
for line in valid:
  user = line[0]
  item = line[1]
  rate = ratings_list[line[2]]
  if user in dic_user_o2i.keys():
    validset.append([dic_user_o2i[user], dic_item_o2i[item], rate])  
for line in test:
  user = line[0]
  item = line[1]
  rate = ratings_list[line[2]]
  if user in dic_user_o2i.keys():
    testset.append([dic_user_o2i[user], dic_item_o2i[item], rate]) 
for line in train:
  user = line[0]
  item = line[1]
  rate = ratings_list[line[2]]
  if user in dic_user_o2i.keys():
    history_u_lists[dic_user_o2i[user]].append(dic_item_o2i[item])
    history_ur_lists[dic_user_o2i[user]].append(rate)
    history_v_lists[dic_item_o2i[item]].append(dic_user_o2i[user])
    history_vr_lists[dic_item_o2i[item]].append(rate)
    data_total.append([dic_user_o2i[user], dic_item_o2i[item], rate])

item_adj_lists = build_item_adj_lists(history_v_lists)

pickle_data = [history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, data_total, validset, testset, social_adj_lists, item_adj_lists, ratings_list]

#different pkl
with open("data/ciao000.pkl", 'wb') as fo:
    pickle.dump(pickle_data, fo)