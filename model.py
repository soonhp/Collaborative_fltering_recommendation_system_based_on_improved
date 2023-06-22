# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 23:34:25 2023

@author: 박순혁
"""

from sklearn.model_selection import StratifiedKFold
from scipy.spatial.distance import squareform, pdist, cdist
from urllib.request import urlopen
from numpy.linalg import norm
from math import isnan, isinf
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle
import json
import gzip
import time
import math
import os
import gc



#%% distance method.
def sim_jac(u,v):
    ind=np.where((1*(u==0)+1*(v==0))==0)[0]
    if len(ind) > 0:
        ind2=np.where((1*(u==0)+1*(v==0))!=2)[0] #합집합
        return len(ind)/ len(ind2)
    else :
        return 0

def sim_cos(u,v):
    
    ind=np.where((1*(u!=0)+1*(v!=0))==2)[0]
    if len(ind) > 0:
        up = sum(u[ind] * v[ind])
        down = norm(u[ind]) * norm(v[ind])
        cos_sim = up/down
        if not math.isnan(cos_sim):
            return cos_sim
        else:
            return 0
    else:
        return 0

def sim_pcc(u,v):
    ind=np.where((1*(u!=0)+1*(v!=0))==2)[0]
    
    if len(ind)>1:
        u_m = np.mean(u[ind])
        v_m = np.mean(v[ind])
        pcc = np.sum((u[ind]-u_m)*(v[ind]-v_m)) / (norm(u[ind]-u_m)*norm(v[ind]-v_m)) # range(-1,1)
        if not isnan(pcc): # case: negative denominator
            return pcc
        else:
            return 0
    else:
        return 0

def sim_msd(u,v):
    ind=np.where((1*(u!=0)+1*(v!=0))==2)[0]
    
    if len(ind)>0:
        msd_sim = 1 - np.sum((u[ind]/5-v[ind]/5)**2)/len(ind)
        if not isnan(msd_sim):
            return msd_sim
        else:
            return 0
    else:
        return 0

    
def sim_cpcc(u,v,r_med=3): # r_med = 3 (CPCC 평균대신 중앙값 사용)
    ind=np.where((1*(u==0)+1*(v==0))==0)[0]
    if len(ind)>1:
        pcc = np.sum((u[ind]-r_med)*(v[ind]-r_med)) / (norm(u[ind]-r_med)*norm(v[ind]-r_med)) # range(-1,1)
        if not isnan(pcc): # case: negative denominator
            return pcc
        else:
            return 0
    else:
        return 0

def sim_spcc(u,v):
    ind=np.where((1*(u==0)+1*(v==0))==0)[0]
    if len(ind)>1:
        u_m = np.mean(u[ind])
        v_m = np.mean(v[ind])
        pcc = np.sum((u[ind]-u_m)*(v[ind]-v_m)) / (norm(u[ind]-u_m)*norm(v[ind]-v_m)) # range(-1,1)
        if not isnan(pcc): # case: negative denominator
            return pcc * (1/(1+np.exp(-len(ind)/2)))
        else:
            return 0
    else:
        return 0

def sim_tmj(u,v):
    ind=np.where((1*(u==0)+1*(v==0))==0)[0]
    if len(ind)>0:
        ind1=np.where((1*(u==0)+1*(v==0))==0)[0]
        ind2=np.where((1*(u==0)+1*(v==0))!=2)[0]
        tri=1-(norm(u[ind]-v[ind]) / (norm(u[ind])+norm(v[ind])))
        return len(ind1)/len(ind2)*tri
    else:
        return 0

def sim_rjac(u,v):
    ind=np.where((1*(u==0)+1*(v==0))==0)[0] # intersection
    if len(ind)>0:
        only_u=len(np.where((1*(u==0)==0))[0]) - len(ind)
        only_v=len(np.where((1*(v==0)==0))[0]) - len(ind)
        return (1/(1+(1/len(ind))+(only_u/(1+only_u))+(1/(1+only_v))))
    else:
        return 0
        
def split_array(x,values=[1,2,3]):
    split_x = []
    for v in values:
        split_x.append(np.where(x==v)[0])   
    return split_x
    
def sim_jacLMH(u,v,interval=np.array([0.1,1.1,5])):
    ind=np.where((1*(u==0)+1*(v==0))==0)[0] # intersection
    if len(ind) > 0:
        u_bins = np.sum(u>=interval[:,None],axis=0)
        v_bins = np.sum(v>=interval[:,None],axis=0)
        u_split = split_array(u_bins)
        v_split = split_array(v_bins)
        uni = np.array([len(np.union1d(i,j)) for i, j in zip(u_split,v_split)])
        inter = np.array([len(np.intersect1d(i,j)) for i, j in zip(u_split,v_split)])
        sel_ind = uni>0
        return np.sum(inter[sel_ind]/uni[sel_ind])/3
    else:
        return 0
        
    
def rating_jaccard(u,v):
    ind=np.where((1*(u==0)+1*(v==0))==0)[0]
    if len(ind) > 0:
        #cnt = sum(np.where(abs(u[ind] -v[ind]) == 0.0, True, False))
        cnt = np.sum((u[ind] -v[ind]) == 0.0)
        return cnt/len(ind)
    else:
        return 0
    

    
def rjac_u(u,v):
    ind=np.where((1*(u==0)+1*(v==0))==0)[0]
    if len(ind) > 0:
        ind2=np.where((1*(u==0)+1*(v==0))!=2)[0] #합집합
        #cnt = sum(np.where(abs(u[ind] -v[ind]) == 0.0, True, False))
        cnt = np.sum((u[ind] -v[ind]) == 0.0)
        return cnt/len(ind2)
    else:
        return 0

    
def rjac_d(u,v):
    global td
    ind=np.where((1*(u==0)+1*(v==0))==0)[0]
    if len(ind) > 0:
        ind2=np.where((1*(u==0)+1*(v==0))!=2)[0] #합집합
        #cnt = sum(np.where(abs(u[ind] - v[ind]) <= td, True, False))
        cnt = np.sum(abs(u[ind] - v[ind]) <= td)
        return cnt/len(ind2)
    else:
        return 0
    
def rjac_dub(u,v):
    global dub_1
    ind=np.where((1*(u==0)+1*(v==0))==0)[0]
    if len(ind) > 0:
        ind2=np.where((1*(u==0)+1*(v==0))!=2)[0] #합집합      
        u_m = np.mean(u[u>=0])
        v_m = np.mean(v[v>=0]) 
        cnt = np.sum(abs((u_m - u[ind]) - (v_m - v[ind])) <= dub_1)
        return cnt/len(ind2)
    else:
        return 0

def wrjac_dub(u,v):
    global dub_1, dub_2
    # dub_1 > dub_2
    ind=np.where((1*(u==0)+1*(v==0))==0)[0]
    if len(ind) > 0:
        ind2=np.where((1*(u==0)+1*(v==0))!=2)[0] #합집합
        #u_ind=np.where((1*(u==0)==0))[0]
        #v_ind=np.where((1*(v==0)==0))[0]
        u_m = np.mean(u[u>=0])
        v_m = np.mean(v[v>=0])
        cuts = np.array([dub_2,dub_1])
        cnt = (abs((u_m - u[ind]) - (v_m - v[ind])) <= cuts[:,None]) * np.array([[0.5],[0.5]])
        return np.sum(cnt)/len(ind2)
    else:
        return 0
#%% 데이터 불러오기 및 rating, item 데이터 전처리.
data_name = 'Netflix'

if data_name == 'MovieLens100K': # MovieLens100K load and preprocessing
    
    td, dub_1, dub_2 = [1.0, 1.0, 0.5]
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
    
    td, dub_1, dub_2 = [1.0, 1.0, 0.5]

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



elif data_name == 'filmtrust': 
    
    td, dub_1, dub_2 = [4/7, 0.2, 0.15]
    
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
    
    td, dub_1, dub_2 = [1.0, 1.4, 0.5]
    
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



#%% 데이터 분할.
# cv validation, random state, split setting.
cv = 5
rs = 35
sk = StratifiedKFold(n_splits=cv, random_state=rs, shuffle=True)

# 결과저장 데이터프레임
result_mae_rmse = pd.DataFrame(columns=['fold','k','MAE','RMSE'])
result_f1 = pd.DataFrame(columns=['fold','k','Precision','Recall','F1_score','fts'])
result_f1_mean = pd.DataFrame(columns=['fold','k','Precision','Recall','F1_score'])
result_cost = pd.DataFrame(columns=['fold','sim_cost','pred_cost','total_cost'])
result_sim_0 = pd.DataFrame(columns=['fold','sim_0'])
result_sim_size = pd.DataFrame(columns=['fold','sim_size'])
count = 0
count2 = 0
count3 = 0
count4 = 0
save_result=dict()
save_f1_fts_result=dict()

sim_dict = {'jac':sim_jac, 'cos':sim_cos, 'pcc':sim_pcc, 'msd':sim_msd, 'cpcc': sim_cpcc, 'spcc': sim_spcc, 'tmj':sim_tmj, 'rating_jaccard':rating_jaccard, 'rjac_u':rjac_u, 'rjac_d':rjac_d, 'rjac_dub':rjac_dub, 'wrjac_dub':wrjac_dub, 'jacLMH':sim_jacLMH, 'rjaccard':sim_rjac}

# 실험.
cross_val=True # cross validation 사용. 
sim_name = 'rating_jaccard'




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
    print(n_item)
    print(n_user)
    
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
    '''
    if sim_name=='cos':    
        sim=pdist(rating_matrix,metric=sim_cos)
        sim=squareform(sim)
    elif sim_name=='pcc':
        sim=pdist(rating_matrix,metric=sim_pcc)
        sim=squareform(sim)
    elif sim_name=='msd':
        sim=pdist(rating_matrix,metric=sim_msd)
        sim=squareform(sim)
    elif sim_name=='cpcc':
        sim=pdist(rating_matrix,metric=sim_cpcc)
        sim=squareform(sim)
    elif sim_name=='spcc':
        sim=pdist(rating_matrix,metric=sim_spcc)
        sim=squareform(sim)
    elif sim_name=='tmj':
        sim=pdist(rating_matrix,metric=sim_tmj)
        sim=squareform(sim)
    elif sim_name=='rating_jaccard':
        sim=pdist(rating_matrix,metric=rating_jaccard)
        sim=squareform(sim)
    elif sim_name=='rjac_u':
        sim=pdist(rating_matrix,metric=rjac_u)
        sim=squareform(sim)
    elif sim_name=='rjac_d':
        sim=pdist(rating_matrix,metric=rjac_d)
        sim=squareform(sim)
    elif sim_name=='rjac_dub':
        sim=pdist(rating_matrix,metric=rjac_dub)
        sim=squareform(sim)        
    elif sim_name=='wrjac_dub':
        sim=pdist(rating_matrix,metric=wrjac_dub)
        sim=squareform(sim)  
    elif sim_name=='jacLMH':
        sim=pdist(rating_matrix,metric=sim_jacLMH)
        sim=squareform(sim)
    '''
    if sim_name!='rjaccard':
        sim=pdist(rating_matrix,metric=sim_dict[sim_name])
        sim=squareform(sim)
    else:
        sim=cdist(rating_matrix,rating_matrix,metric=sim_dict[sim_name])        
        
    print(time.time()-s)
    sc_cost = time.time()-s

    # sel_nn, sel_sim: neighbor 100 명까지만 id와 similarity를 저장.
    np.fill_diagonal(sim,-1)
    sim_0 = np.count_nonzero(sim==0)/2
    sim_size = (sim.size-sim.shape[0])/2
    result_sim_0.loc[count4] = [f,sim_0]
    result_sim_size.loc[count4] = [f,sim_size]
    count4 += 1
    nb_ind=np.argsort(sim,axis=1)[:,::-1] # nearest neighbor sort.
    sel_nn=nb_ind[:,:100]
    sel_sim=np.sort(sim,axis=1)[:,::-1][:,:100]


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

        result_mae_rmse.loc[count] = [f, k, mae, rmse]

        pp = np.array(pp)
        rr = np.array(rr)
        mm = np.array(mm)
###
        if data_name == 'filmtrust' :
            f1_thr_score = [23/7, 27/7, 31/7]
        else :
            f1_thr_score = [3.5, 4, 4.5]


        for fts in f1_thr_score :
            TPP = len(set(np.where(pp >= fts)[0]).intersection(set(np.where(rr >= fts)[0])))
            FPP = len(set(np.where(pp >= fts)[0]).intersection(set(np.where(rr < fts)[0])))
            FNP = len(set(np.where(pp < fts)[0]).intersection(set(np.where(rr >=fts)[0])))
            _precision = TPP / (TPP + FPP)
            _recall = TPP / (TPP + FNP)
            _f1_score = 2 * _precision * _recall / (_precision + _recall)
            result_f1.loc[count2] = [f, k, _precision, _recall, _f1_score, fts]


            count2 += 1
        # precision, recall, f1-score
###
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
    total_cost = sc_cost + p_cost
    result_cost.loc[count3]=[f,sc_cost,p_cost,total_cost]
    count3 += 1
    
    # 반복여부 (cross validation)
    if cross_val == True:
        continue
    else:
        break

#%%
result_1 = result_mae_rmse.groupby(['k']).mean().drop(columns=['fold'])
result_2 = result_f1_mean.groupby(['k']).mean().drop(columns=['fold','Precision','Recall'])
result = pd.merge(result_1, result_2, on=result_1.index).drop(columns=['key_0'])
result_3 = result_f1.groupby(['k','fts']).mean().drop(columns=['fold'])
result_fts = result_3.copy()
result_time_cost = result_cost.drop(columns=['fold'])
cost = result_cost['total_cost'].mean()
print('Time cost : ', cost)
result_4 = result_sim_0['sim_0'].mean()
result_5 = result_sim_size['sim_size'].mean()
result_6 = result_4/result_5   # Number of Similarity Zeros (Ratio)
sim_0_dict = [{'sim_0':result_4,'sim_size':result_5,'ratio':result_6}]
sim_0_ratio = pd.DataFrame(sim_0_dict)
result_fts_35 = result_fts.xs(3.5,level='fts')
#final_dict[sim_name] = result_7.copy()
# %% 시험결과 저장.
import datetime
result.to_csv('result/{}_{}_{}_result.csv'.format(data_name,sim_name,str(datetime.datetime.now())[:13]+'시'+str(datetime.datetime.now())[14:16]+'분'))
result_fts.to_csv('result/{}_{}_{}_result_fts.csv'.format(data_name,sim_name,str(datetime.datetime.now())[:13]+'시'+str(datetime.datetime.now())[14:16]+'분'))
result_fts_35.to_csv('result/{}_{}_{}_result.csv'.format(data_name,sim_name,str(datetime.datetime.now())[:13]+'시'+str(datetime.datetime.now())[14:16]+'분'))
result_time_cost.to_csv('result/{}_{}_{}_result.csv'.format(data_name,sim_name,str(datetime.datetime.now())[:13]+'시'+str(datetime.datetime.now())[14:16]+'분'))
sim_0_ratio.to_csv('result/{}_{}_{}_result.csv'.format(data_name,sim_name,str(datetime.datetime.now())[:13]+'시'+str(datetime.datetime.now())[14:16]+'분'))

