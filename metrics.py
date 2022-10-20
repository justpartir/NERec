import math
import pandas as pd
import numpy as np

cut_off = 4

class MetronAtK(object):
    def __init__(self, top_k):
        self._top_k = top_k
        self._subjects = None  # Subjects which we ran evaluation on
        self.eps = 1e-10

    @property
    def top_k(self):
        return self._top_k

    @top_k.setter
    def top_k(self, top_k):
        self._top_k = top_k

    @property
    def subjects(self):
        return self._subjects

    @subjects.setter
    def subjects(self, subjects):
        """
        args:
            subjects: list, [test_users, test_items, test_scores, negative users, negative items, negative scores]
        """
        assert isinstance(subjects, list)
        users, items, ratings, scores, labels = subjects[0], subjects[1], subjects[2], subjects[3], subjects[4]
        # the full set
        full = pd.DataFrame({'user': users,
                            'item': items,
                            'rating': ratings,
                             'score': scores,
                             'label': labels})
        # rank the items according to the scores for each user
        # full['score'] = full.apply(lambda x: math.log(x.score)-math.log(1-x.score), axis=1)
        full['label_predicted'] = full.apply(lambda x: 1 if x.score>=0.5 else 0, axis=1)
        full['rank_truth'] = full.groupby('user')['rating'].rank(method='first', ascending=False)
        full['rank_predicted'] = full.groupby('user')['score'].rank(method='first', ascending=False)
        self._subjects = full

    def cal_recall(self):
        """召回率 top_K"""
        recall_list = []
        full, top_k = self._subjects, self._top_k
        top_k = full[full['rank_predicted']<=top_k]
        top_k['count'] = top_k.apply(lambda x: 1 if x.label==1 and x.label==x.label_predicted else 0, axis=1)
        for u in top_k['user'].unique():
            recall_u = top_k[top_k['user']==u]['count'].sum()/(top_k[top_k['user']==u]['label'].sum() + self.eps)
            recall_list.append(recall_u)
        return np.mean(recall_list)

    def cal_ndcg(self):
        """命中率 top_K"""
        ndcg = []
        full, top_k = self._subjects, self._top_k
        top_k_pre = full[full['rank_predicted']<=top_k]
        top_k_pre.sort_values(['user', 'rank_predicted'], inplace=True)
        top_k_pre['dcg'] = top_k_pre.apply(lambda x: x.label/math.log2(x.rank_predicted+1), axis=1)
        top_k_tru = full[full['rank_truth']<=top_k]
        top_k_tru['idcg'] = top_k_tru.apply(lambda x: x.label/math.log2(x.rank_truth+1), axis=1)
        top_k_tru.sort_values(['user', 'rank_truth'], inplace=True)
        for u in top_k_pre['user'].unique():
            ndcg_u = top_k_pre[top_k_pre['user']==u]['dcg'].sum()/(top_k_tru[top_k_tru['user']==u]['idcg'].sum() + self.eps)
            ndcg.append(ndcg_u)
        return np.mean(ndcg)
