#!/usr/bin/env python
# coding: utf-8

# In[26]:


get_ipython().run_line_magic('run', 'round2_base.ipynb')
import lightgbm as lgb
import random


# In[27]:


def drop_features(df, feature):
    drop_cols = [col for col in feature.columns if col not in ['buyer_admin_id', 'item_id', 'cate_id', 'store_id']]
    print('> drop features number is ', len(drop_cols))
    return df.drop(drop_cols, 1)

def train_check(df):
    dupli_columns = df.columns.duplicated()
    if sum(dupli_columns) > 0:
        raise ValueError('columns:{} is duplicated'.format(' ,'.join(df.columns[dupli_columns])))

def train_split(df, label='irank'):
    train_check(df)
    df['country_id'] = df['country_id'].map({'xx':0, 'yy':1, 'zz':2}).astype('category')
    df['cate_id'] = df['cate_id'].astype('category')
    df['store_id'] = df['store_id'].astype('category')
    train_df = df[df['is_train'] == 1].reset_index(drop=True)
    test_df = df[(df['is_train'] == 0) & (df['data_type']=='base')].reset_index(drop=True)
    print('> train_df sample:', len(train_df))
    print('>> positive-1 sample:', len(train_df[train_df[label]==1]))
    print('>> negtive-1 sample:', len(train_df[train_df[label]==0]))
    print('>> 0/1 sample:', len(train_df[train_df[label]==0]) / len(train_df[train_df[label]==1]))
    print('> features number is:', len(train_df.columns))
    print('> test_df sample:', len(test_df),'\n')
    return train_df, test_df

def get_lgb_params():
    learning_rate = 0.1
    objective = 'binary'
    lgb_params = {
        'num_leaves': 127, #31
        'min_data_in_leaf': 15, # 30 
        'objective':objective,
        'max_depth': -1,
        'learning_rate': learning_rate,
        "min_child_samples": 15,
        "boosting": "gbdt",
        "feature_fraction": 0.8,
        "bagging_freq": 1,
        "bagging_fraction": 0.9 ,
        "bagging_seed": 11,
        "metric": 'auc',
        "lambda_l1": 0.1,
        "verbosity": -1,
        "nthread": 23,
        "random_state": 4590,
         }
    return lgb_params

def train_evaluation(df):
    # label
    label = pd.read_hdf('../data/label.h5', '1.0')
    label = label[label['buyer_admin_id'].isin(df['buyer_admin_id'])]
    
    user_num = len(df)
    df = df.sort_values(by=['buyer_admin_id', 'irank'], ascending=False)
    df['irank'] = df.groupby(['buyer_admin_id']).cumcount() + 1    
    df = pd.merge(df, label, how='inner', on=['buyer_admin_id', 'item_id'])
    
    score_df = pd.DataFrame(df['irank'].value_counts().head(10)/user_num)
    score_df['cum_irank'] = score_df['irank'].cumsum()
    print(score_df.head(20))
    
    score = sum(1 / score_df['irank']) / user_num
    print('evaluation score: ',score)
    return score_df, score

def train_user_set(df, n=5):
    user_set = df[(df['country_id']!=0)  & (df['data_type']=='base')][['buyer_admin_id']].drop_duplicates()
    user_set = user_set.sample(frac=1, random_state=2020)['buyer_admin_id'].unique()
    user_set = [user_set[i::n] for i in range(n)]
    return user_set
    

def train_kfold_lgb(features, if_online=None, label='irank', exclude_columns=None, categorical_columns=None, return_valid=False):
    train_df, test_df = train_split(features)
    features_columns = [f for f in train_df.columns if f not in exclude_columns]
    
    # Create arrays and dataframes to store results
    label_df = train_df[(train_df['irank']==1) & (train_df['data_type']=='base')][['buyer_admin_id', 'item_id']]
    oof_preds = train_df[['buyer_admin_id', 'item_id']]
    sub_preds = np.zeros(test_df.shape[0])
    mean_score = 0
    mean_baseline_score = 0
    user_set = train_user_set(train_df, n=5)

    for n_fold, valid_user in enumerate(user_set):
        train_idx = train_df[~train_df['buyer_admin_id'].isin(valid_user)] # add slide user
        valid_idx = train_df[(train_df['buyer_admin_id'].isin(valid_user)) & (train_df['data_type']=='base')]
        
        xgtrain = lgb.Dataset(train_idx[features_columns], label=train_idx[label].values, categorical_feature = categorical_columns)
        del train_idx; gc.collect()
        xgvalid = lgb.Dataset(valid_idx[features_columns], label=valid_idx[label].values, categorical_feature = categorical_columns)
        lgb_params = get_lgb_params()
        clf = lgb.train(lgb_params,
                         xgtrain,
                         valid_sets=[xgtrain,xgvalid],
                         valid_names=['train','valid'],
                         num_boost_round=10000,
                         early_stopping_rounds=100, 
                         verbose_eval = 500,
                         )
        
        # 结果处理
        user_num = len(valid_user)
        valid_idx['proba'] = clf.predict(valid_idx[features_columns], num_iteration=clf.best_iteration)
        
        try:
            valid_idx.sort_values(by=['buyer_admin_id', 'baseline'], ascending=[1, 1], inplace=True)
        except:
            valid_idx.sort_values(by=['buyer_admin_id', 'user-item_without_repeat_irank_without_repeat_MIN'], ascending=[1, 1], inplace=True)
        
        valid_idx['baseline_rank'] = valid_idx.groupby(['buyer_admin_id']).cumcount() + 1
        
        valid_idx.sort_values(by=['buyer_admin_id', 'proba'], ascending=[1, 0], inplace=True)
        valid_idx['proba_rank'] = valid_idx.groupby(['buyer_admin_id']).cumcount() + 1
        
        
        score_df = pd.merge(label_df, 
                            valid_idx[['buyer_admin_id', 'item_id', 'baseline_rank', 'proba_rank', 'proba']], 
                            how='inner', on=['buyer_admin_id', 'item_id'])
        
        proba_score = sum(1 / score_df['proba_rank']) / user_num
        baseline_score = sum(1 / score_df['baseline_rank']) / user_num
        
        mean_score += proba_score
        mean_baseline_score += baseline_score
        
        score_ratio_df = pd.concat([score_df['proba_rank'].value_counts(),
                                   score_df['baseline_rank'].value_counts()], axis=1) / user_num
        
        print(score_ratio_df.head(5))
        print('evaluation baseline score: ',baseline_score)
        print('evaluation proba score: ', proba_score)
        
        
        if if_online:
            sub_preds += clf.predict(test_df[features_columns], num_iteration=clf.best_iteration) # folds.n_splits
    
    print('-'*40, '\n', 'mean evaluation baseline score: ', mean_baseline_score/ (n_fold + 1))
    print('mean evaluation proba score: ',mean_score/ (n_fold + 1), '\n')
    
    if return_valid:
        return valid_idx
    
    if if_online:
        test_df[label] = sub_preds / (n_fold + 1)
        submit = test_df[['buyer_admin_id', 'item_id', 'irank']]
        submit = submit.sort_values(by=['buyer_admin_id', 'irank'], ascending=[1,0]).reset_index(drop=True)
        submit['irank'] = submit.groupby(['buyer_admin_id']).cumcount() + 1
        return submit, mean_score
    else:
        return score_df, mean_score


# In[28]:


def get_features(data_type):
    if data_type == 'base':
        path = '../feature/rank/'
    elif data_type == 'slide':
        path = '../feature/slide/'
    
    print('> Now dtype is :', data_type)
    features = get_sample(dtype='rank', data_type=data_type)
    features['data_type'] = data_type

    path_tuple = [
        ('user_item_dedup_feature', 'asc'),
        ('user_item_dedup_feature', 'cate_id_desc'),
        ('user_item_dedup_feature', 'store_id_desc'),
        ('user_item_dupli_feature', 'all'),
        ('user_second_diff_feature', 'item_id'),
        ('user_second_diff_feature', 'cate_id'),
        ('user_second_diff_feature', 'store_id'),
        ('user_cate_lastday_dedup_feature', 'all'),
        ('user_item_lastday_dedup_feature', 'all'),
        ('user_store_lastday_dedup_feature', 'all'),
        ('user_cate_dedup_feature', 'all'),
        ('user_item_dedup_feature', 'all'),
        ('user_store_dedup_feature', 'all'),
        ('user_item_rank_diff_feature', 'all'),
        ('user_cate_dedup_feature', 'asc'),
        ('user_store_dedup_feature', 'asc'),
        ('user_item_lastday_dedup_feature', 'desc'),
        ('user_item_lastday_dedup_feature', 'asc'),
        ('item_conv_feature', 'all'),
        ('user_cate_feature', 'buy'),
    ]

    for file in path_tuple:
        print(file)
        feature = pd.read_hdf(path + file[0], file[1])
        if file[0] == 'item_conv_feature':
            key = ['item_id']
        elif 'item_id' in feature.columns:
            key = ['buyer_admin_id', 'item_id']
            feature = feature.drop([col for col in feature.columns if col in ['cate_id','store_id']], 1)
        else:
            key = [col for col in feature.columns if col in ['buyer_admin_id', 'cate_id', 'store_id']]
        features = features.merge(feature, on=key, how='left')
        
    return features


# In[13]:


base_features = get_features('base')


# In[6]:


submit, score = train_kfold_lgb(base_features.copy(),
                                if_online=True,
                                label='irank',
                                exclude_columns=['buyer_admin_id', 'item_id', 'irank', 'is_train', 'baseline', 'data_type'], 
                                categorical_columns=['cate_id', 'store_id', 'country_id'])


# In[ ]:


# def get_submitfile():
# 品类填充
recall = pd.read_hdf('../output/recall_by_next_5_item', 'all')
submit_recall = pd.concat([submit1, recall])
submit_recall['irank'] = submit_recall.groupby(['buyer_admin_id']).cumcount()+1
submit_file  = submit_transform(submit_recall, '0907-01.csv')

