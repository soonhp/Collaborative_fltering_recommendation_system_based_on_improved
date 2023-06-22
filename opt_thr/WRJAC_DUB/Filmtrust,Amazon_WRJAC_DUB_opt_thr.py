# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 15:38:07 2023

@author: 박순혁
"""

from sklearn.model_selection import StratifiedKFold
from scipy.spatial.distance import squareform, pdist
from numpy.linalg import norm
from math import isnan, isinf
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle
import time
import math
import os

#%% distance method.


def wrjac_dub(u,v):
    global td
    ind=np.where((1*(u==0)+1*(v==0))==0)[0]
    if len(ind) > 0:
        ind2=np.where((1*(u==0)+1*(v==0))!=2)[0] #합집합
        u_ind=np.where((1*(u==0)==0))[0]
        v_ind=np.where((1*(v==0)==0))[0]
        u_m = np.mean(u[u_ind])
        v_m = np.mean(v[v_ind])
        cnt = sum(np.where(abs((u_m - u[ind]) - (v_m - v[ind])) <= td, True, False))
        cnt2 = sum(np.where((abs((u_m - u[ind]) - (v_m - v[ind])) > td) & (abs((u_m - u[ind]) - (v_m - v[ind])) <= 0.2), True, False))
        cnt3 = sum(np.where(abs(u[ind] -v[ind]) == 0.0, True, False))
        if abs(u_m - v_m) <= 0.3 or cnt3 == len(ind) :
            cnt4 = 1
        else:
            cnt4 = 0
        return (cnt+(cnt2*0.5)+cnt4)/len(ind2)
    else:
        return 0
    



#%% 데이터 불러오기 및 rating, item 데이터 전처리.
data_name = 'filmtrust'

if data_name == 'MovieLens100K': # MovieLens100K load and preprocessing
    
    # MovieLens100K: u.data, item.txt 의 경로
    data = pd.read_table('D:/collaborative_filtering/movielens/order/u.data',header=None, names=['uid','iid','r','ts'])
    data = data.drop(columns=['ts'])
    user = data.drop_duplicates(['uid']).reset_index(drop=True)
    item_data = data.drop_duplicates(['iid']).reset_index(drop=True)

    m_d = {}
    for n, i in enumerate(item_data.iloc[:,1]):
        m_d[i] = n
    item_data.iloc[:,0] = sorted(m_d.values())

    i_to_n = []
    for i in range(data.shape[0]):
        i_to_n.append(m_d[data.loc[i,'iid']])
    data['iid'] = i_to_n    


    u_d = {}
    for n, i in enumerate(user.iloc[:,0]):
        u_d[i] = n
    user.iloc[:,0] = sorted(u_d.values())

    u_to_n = []
    for u in range(data.shape[0]):
        u_to_n.append(u_d[data.loc[u,'uid']])
    data['uid'] = u_to_n  


    
elif data_name == 'MovieLens1M': 
    
    # MovieLens100K: u.data, item.txt 의 경로
    data = pd.read_csv('D:/ml-1m/ratings.dat', sep='::', names=['uid','iid','r','ts'], encoding='latin-1',header=None)
    data = data.drop(columns=['ts'])
    user = data.drop_duplicates(['uid']).reset_index(drop=True)
    item_data = data.drop_duplicates(['iid']).reset_index(drop=True)

    m_d = {}
    for n, i in enumerate(item_data.iloc[:,1]):
        m_d[i] = n
    item_data.iloc[:,0] = sorted(m_d.values())

    i_to_n = []
    for i in range(data.shape[0]):
        i_to_n.append(m_d[data.loc[i,'iid']])
    data['iid'] = i_to_n    


    u_d = {}
    for n, i in enumerate(user.iloc[:,0]):
        u_d[i] = n
    user.iloc[:,0] = sorted(u_d.values())

    u_to_n = []
    for u in range(data.shape[0]):
        u_to_n.append(u_d[data.loc[u,'uid']])
    data['uid'] = u_to_n  


elif data_name == 'movietweetings':
    
    movietw_data=pd.read_csv('ratings.dat',sep='::', names=['uid','iid','r','ts'], engine='python')
    movietw_data=movietw_data.drop(columns=['ts'])
    gb_inum = movietw_data[['uid','iid']].groupby(['uid']).count()
    over_20_idxs = gb_inum.loc[gb_inum.iid > 20].index.values
    data = movietw_data.loc[movietw_data.uid.isin(over_20_idxs)].reset_index(drop=True)
    change_r_dict = {1.0:1.0, 2.0:13/9, 3.0:17/9, 4.0:21/9, 5.0:25/9, 6.0:29/9, 7.0:33/9 ,8.0:37/9, 9.0:41/9, 10.0:5.0}
    data=data.replace({'r':change_r_dict})
    data['r']= round(data['r'],2)
    user = data.drop_duplicates(['uid']).reset_index(drop=True)
    item_data = data.drop_duplicates(['iid']).reset_index(drop=True)

    m_d = {}
    for n, i in enumerate(item_data.iloc[:,1]):
        m_d[i] = n
    item_data.iloc[:,0] = sorted(m_d.values())

    i_to_n = []
    for i in range(data.shape[0]):
        i_to_n.append(m_d[data.loc[i,'iid']])
    data['iid'] = i_to_n    


    u_d = {}
    for n, i in enumerate(user.iloc[:,0]):
        u_d[i] = n
    user.iloc[:,0] = sorted(u_d.values())

    u_to_n = []
    for u in range(data.shape[0]):
        u_to_n.append(u_d[data.loc[u,'uid']])
    data['uid'] = u_to_n   


elif data_name == 'filmtrust': 
    # Netflix: ratings.csv, movies.csv 의 경로
    flimtrust_data=pd.read_table('D:/filmtrust/ratings.txt', sep=' ', names=['uid','iid','r'])
    gb_inum = flimtrust_data[['uid','iid']].groupby(['uid']).count()
    over_20_idxs = gb_inum.loc[gb_inum.iid > 20].index.values
    data = flimtrust_data.loc[flimtrust_data.uid.isin(over_20_idxs)].reset_index(drop=True)
    change_r_dict = {0.5:1.0, 1.0:11/7, 1.5:15/7, 2.0:19/7, 2.5:23/7, 3.0:27/7, 3.5:31/7 ,4.0:5.0}
    data=data.replace({'r':change_r_dict})
    data['r']= round(data['r'],2)
    user = data.drop_duplicates(['uid']).reset_index(drop=True)
    item_data = data.drop_duplicates(['iid']).reset_index(drop=True)


    m_d = {}
    for n, i in enumerate(item_data.iloc[:,1]):
        m_d[i] = n
    item_data.iloc[:,0] = sorted(m_d.values())

    i_to_n = []
    for i in range(data.shape[0]):
        i_to_n.append(m_d[data.loc[i,'iid']])
    data['iid'] = i_to_n    


    u_d = {}
    for n, i in enumerate(user.iloc[:,0]):
        u_d[i] = n
    user.iloc[:,0] = sorted(u_d.values())

    u_to_n = []
    for u in range(data.shape[0]):
        u_to_n.append(u_d[data.loc[u,'uid']])
    data['uid'] = u_to_n   

    

elif data_name == 'CiaoDVD': 
    # Netflix: ratings.csv, movies.csv 의 경로
    ciaodvd_data=pd.read_table('D:/CiaoDVD/movie-ratings.txt', sep=',', names=['uid','iid','gid','rid','r','ts'])
    ciaodvd_data=ciaodvd_data.drop(columns=['gid','rid','ts'])
    gb_inum = ciaodvd_data[['uid','iid']].groupby(['uid']).count()
    over_20_idxs = gb_inum.loc[gb_inum.iid > 20].index.values
    data = ciaodvd_data.loc[ciaodvd_data.uid.isin(over_20_idxs)].reset_index(drop=True)
    user = data.drop_duplicates(['uid']).reset_index(drop=True)
    item_data = data.drop_duplicates(['iid']).reset_index(drop=True)


    m_d = {}
    for n, i in enumerate(item_data.iloc[:,1]):
        m_d[i] = n
    item_data.iloc[:,0] = sorted(m_d.values())

    i_to_n = []
    for i in range(data.shape[0]):
        i_to_n.append(m_d[data.loc[i,'iid']])
    data['iid'] = i_to_n    


    u_d = {}
    for n, i in enumerate(user.iloc[:,0]):
        u_d[i] = n
    user.iloc[:,0] = sorted(u_d.values())

    u_to_n = []
    for u in range(data.shape[0]):
        u_to_n.append(u_d[data.loc[u,'uid']])
    data['uid'] = u_to_n   

elif data_name == 'Amazon':
    
    data=pd.read_csv('AmazonMovie_small.csv')
    
elif data_name == 'Netflix':
    
    data=pd.read_csv('Netflix_small.csv')
    print(data.isnull().sum())

#%%
# Collaborative Filtering
##########################################################################################################################################################
##########################################################################################################################################################
##########################################################################################################################################################
# m -> parameter
td_list = [0.05,0.1,0.15]

#%% 데이터 분할.
# cv validation, random state, split setting.
cv = 5
rs = 35
sk = StratifiedKFold(n_splits=cv, random_state=rs, shuffle=True)

# 결과저장 데이터프레임
save_result=dict()
save_f1_fts_result=dict()

# 실험.
cross_val=True # cross validation 사용. 

sim_name = 'wrjac_dub'

for td in td_list :
    # 결과저장 데이터프레임
    result_mae_rmse = pd.DataFrame(columns=['fold','td','k','MAE','RMSE'])
    result_f1 = pd.DataFrame(columns=['fold','td','k','Precision','Recall','F1_score','fts'])
    result_f1_mean = pd.DataFrame(columns=['fold','k','Precision','Recall','F1_score'])
    result_cost = pd.DataFrame(columns=['fold','cost'])
    count = 0
    count2 = 0
    count3 = 0
    count4 = 0
    # split dataset
    for f, (trn,val) in enumerate(sk.split(data,data['uid'].values)):
        print()
        print(f'cv: {f+1}')
        trn_data = data.iloc[trn]
        val_data = data.iloc[val]

        # train dataset rating dictionary.
        data_d_trn_data = {}
        for u, i, r in zip(trn_data['uid'], trn_data['iid'], trn_data['r']):
            if u not in data_d_trn_data:
                data_d_trn_data[u] = {i:r}
            else:
                data_d_trn_data[u][i] = r

        # train dataset user rating mean dictionary.
        data_d_trn_data_mean = {}
        for u in data_d_trn_data:
            data_d_trn_data_mean[u] = np.mean(list(data_d_trn_data[u].values()))


        #%% rating matrix about train/test set.

        n_item = len(set(data['iid']))
        n_user = len(set(data['uid']))

        # train rating matrix
        rating_matrix = np.zeros((n_user, n_item))
        for u, i, r in zip(trn_data['uid'], trn_data['iid'], trn_data['r']):
            rating_matrix[u,i] = r

        # test rating matrix
        rating_matrix_test = np.zeros((n_user, n_item))
        for u, i, r in zip(val_data['uid'], val_data['iid'], val_data['r']):
            rating_matrix_test[u,i] = r



        #%% 1. similarity calculation.

        print('\n')
        print(f'similarity calculation: {sim_name}')

        s=time.time()


        # 기본적인 유사도지표
        if sim_name=='wrjac_dub':    
            sim=pdist(rating_matrix,metric=wrjac_dub)
            sim=squareform(sim)


        print(time.time()-s)
        sc_cost = time.time()-s
        result_cost.loc[count4]=[f,sc_cost]

        # sel_nn, sel_sim: neighbor 100 명까지만 id와 similarity를 저장.
        np.fill_diagonal(sim,-1)
        nb_ind=np.argsort(sim,axis=1)[:,::-1] # nearest neighbor sort.
        sel_nn=nb_ind[:,:100]
        sel_sim=np.sort(sim,axis=1)[:,::-1][:,:100]
        count4 += 1

        #%% 2. prediction
        print('\n')
        print('prediction: k=10,20, ..., 100')
        rating_matrix_prediction = rating_matrix.copy()

        s=time.time()

        for k in tqdm([10,20,30,40,50,60,70,80,90,100]):

            for user in range(rating_matrix.shape[0]):

                for p_item in list(np.where(rating_matrix_test[user,:]!=0)[0]):

                    molecule = []
                    denominator = []

                    #call K neighbors
                    user_neighbor = sel_nn[user,:k]
                    user_neighbor_sim = sel_sim[user,:k]

                    for neighbor, neighbor_sim in zip(user_neighbor, user_neighbor_sim):

                        if p_item in data_d_trn_data[neighbor].keys():
                            molecule.append(neighbor_sim * (rating_matrix[neighbor, p_item] - data_d_trn_data_mean[neighbor]))
                            denominator.append(abs(neighbor_sim))
                    try:
                        rating_matrix_prediction[user, p_item] = data_d_trn_data_mean[user] + (sum(molecule) / sum(denominator))
                    except ZeroDivisionError:
                        rating_matrix_prediction[user, p_item] = math.nan




           #%%3. performance
            # MAE, RMSE

            precision, recall, f1_score = [], [], []
            pp=[]
            rr=[]
            mm=[]

            for u, i, r in zip(val_data['uid'], val_data['iid'], val_data['r']):
                p = rating_matrix_prediction[u,i]
                um = data_d_trn_data_mean[u]
                if not math.isnan(p):
                    pp.append(p)
                    rr.append(r)
                    mm.append(um)

            d = [abs(a-b) for a,b in zip(pp,rr)]
            mae = sum(d)/len(d)
            rmse = np.sqrt(sum(np.square(np.array(d)))/len(d))

            result_mae_rmse.loc[count] = [f, td, k, mae, rmse]

            pp = np.array(pp)
            rr = np.array(rr)
            mm = np.array(mm)
    ###

            f1_thr_score = [23/7, 27/7, 31/7]


            for fts in f1_thr_score :
                TPP = len(set(np.where(pp >= fts)[0]).intersection(set(np.where(rr >= fts)[0])))
                FPP = len(set(np.where(pp >= fts)[0]).intersection(set(np.where(rr < fts)[0])))
                FNP = len(set(np.where(pp < fts)[0]).intersection(set(np.where(rr >=fts)[0])))
                _precision = TPP / (TPP + FPP)
                _recall = TPP / (TPP + FNP)
                _f1_score = 2 * _precision * _recall / (_precision + _recall)
                result_f1.loc[count2] = [f, td, k, _precision, _recall, _f1_score, fts]

            count2 += 1
            TPP = len(set(np.where(pp >= mm)[0]).intersection(set(np.where(rr >= mm)[0])))
            FPP = len(set(np.where(pp >= mm)[0]).intersection(set(np.where(rr < mm)[0])))
            FNP = len(set(np.where(pp < mm)[0]).intersection(set(np.where(rr >=mm)[0])))
            _precision = TPP / (TPP + FPP)
            _recall = TPP / (TPP + FNP)
            _f1_score = 2 * _precision * _recall / (_precision + _recall)
            result_f1_mean.loc[count] = [f, k, _precision, _recall, _f1_score]
            
            count += 1
        print(time.time() - s)
        p_cost = time.time()-s

        count3 += 1
        
        # 반복여부 (cross validation)
        if cross_val == True:
            continue
        else:
            break

    #%%
    result_1 = result_mae_rmse.groupby(['k']).mean().drop(columns=['fold'])
    save_result[td]=result_1.drop(columns=['td'])
    result_2 = result_f1_mean.groupby(['k']).mean().drop(columns=['fold','Precision','Recall'])
    
    result = pd.merge(result_1, result_2, on=result_1.index).drop(columns=['key_0'])

    #%%
    result_1 = result_mae_rmse.groupby(['k']).mean().drop(columns=['fold'])
    save_result[m]=result_1.drop(columns=['td'])
    result_2 = result_f1_mean.groupby(['k']).mean().drop(columns=['fold','Precision','Recall'])
    result = pd.merge(result_1, result_2, on=result_1.index).drop(columns=['key_0'])
    result_3 = result_f1.groupby(['k','fts']).mean().drop(columns=['fold'])
    save_f1_fts_result[td] = result_3.drop(columns=['td'])
    result_fts = result_3.copy()
    
    cost = result_cost['cost'].mean()
    print('Time cost : ', cost)

    #%% 시험결과 저장.
    import datetime
    result.to_csv('{}_{}_{}_{}_result.csv'.format(data_name,sim_name,td,str(datetime.datetime.now())[:13]+'시'+str(datetime.datetime.now())[14:16]+'분'))
    result_fts.to_csv('{}_{}_{}_{}_result_fts.csv'.format(data_name,sim_name,td,str(datetime.datetime.now())[:13]+'시'+str(datetime.datetime.now())[14:16]+'분'))



