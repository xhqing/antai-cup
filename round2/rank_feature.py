#!/usr/bin/env python
# coding: utf-8

# ### Tips
# * 生成特征时，一定要记得过滤label `df = get_hdf('all', if_filter_label=True, if_drop_duplicates=True)`
# * 分布式运行pandas `import modin.pandas as pd`, 必须放到读取文件之后
# * 连续使用同一个df创建特征时， 给函数传递参数时必须加上`df.copy`， 否则`df`在函数内会被改变

# In[12]:


# %run round2_base.ipynb
# from round2_base import *


# In[13]:


# sample = get_sample(dtype='rank')
# df = get_hdf('all', if_filter_label=True, if_drop_duplicates=True, if_lastday=True)


# In[2]:


def add_prefix(df, exclude_columns, prefix):
    if isinstance(exclude_columns, str):
        exclude_columns = [exclude_columns]
        
    column_names = [col for col in df.columns if col not in exclude_columns]
    df.rename(columns = dict(zip(column_names, [prefix + name for name in column_names])), inplace=True)
    return df


# In[3]:


def group_func(df, group_func_dic, group_key):
    if isinstance(group_func_dic, str):
        group_func_dic = [group_func_dic]
        
    features = df.groupby(group_key).agg(group_func_dic)
    features.columns = [e[0] + "_" + e[1].upper() for e in features.columns.tolist()]
    features.reset_index(inplace=True)
    return features


# In[4]:


def filter_sample(df, key=None):
    if key is None:
        df = df.merge(sample[['buyer_admin_id']].drop_duplicates(), on=['buyer_admin_id'], how='inner')
    else:
        df = df.merge(sample[['buyer_admin_id', key]].drop_duplicates(), on=['buyer_admin_id', key], how='inner')
    return df


# In[5]:


def get_user_item_dupli_feature(df):
    """
    df = get_hdf('buy', if_filter_label=True)
    import modin.pandas
    get_user_item_dupli_feature(df.copy())
    """
    user = df[df[['buyer_admin_id', 'item_id', 'log_time']].duplicated(keep=False)][['buyer_admin_id','item_id']].drop_duplicates()
    dup = df.merge(user, how='inner', on=['buyer_admin_id', 'item_id'])
    
    feature_type = {
        'dense_rank' : ['max', 'min', np.ptp],
        'first_second_diff':['max', 'min', 'mean'],
        'last_second_diff':['max', 'min', 'mean']
    }    
    dup_feature = group_func(dup, feature_type, group_key=['buyer_admin_id', 'item_id'])
    dup_feature = add_prefix(dup_feature, ['buyer_admin_id', 'item_id'], 'user_item_dup_')
    
    dup_cnt = dup.groupby(['buyer_admin_id', 'item_id', 'log_time']).size().to_frame('dup_cnt').reset_index()

    feature_type = {
        'dup_cnt':['first', 'max', 'min', 'last', 'nunique'],
    }
    
    feature = group_func(dup_cnt, feature_type, group_key=['buyer_admin_id', 'item_id'])
    feature = add_prefix(feature, ['buyer_admin_id', 'item_id'], 'user-item_')
    feature['user-item_dup_cnt_FIRST=MAX'] = feature['user-item_dup_cnt_MAX'] - feature['user-item_dup_cnt_FIRST']
    
    irank2 = df[df['irank']==2][['buyer_admin_id', 'item_id']]
    irank2_flag = irank2.merge(dup_cnt[['buyer_admin_id', 'item_id']].drop_duplicates(), how='inner', on=['buyer_admin_id', 'item_id'])        .merge(feature, how='left', on=['buyer_admin_id', 'item_id'])
    irank2_flag['irank2_is_dup'] = 1
    irank2_flag['irank2_is_dup_scope'] = irank2_flag['irank2_is_dup'] * (irank2_flag['user-item_dup_cnt_FIRST'] < irank2_flag['user-item_dup_cnt_MAX'])
    irank2_flag = irank2_flag.drop([col for col in irank2_flag.columns if 'user-item' in col], 1)

    irank3 = df[df['irank']==3][['buyer_admin_id', 'item_id']]
    irank3_flag = irank3.merge(dup_cnt[['buyer_admin_id', 'item_id']].drop_duplicates(), how='inner', on=['buyer_admin_id', 'item_id'])        .merge(feature, how='left', on=['buyer_admin_id', 'item_id'])
    irank3_flag['irank3_is_dup'] = 1
    irank3_flag['irank3_is_dup_scope'] = irank3_flag['irank3_is_dup'] * (irank3_flag['user-item_dup_cnt_FIRST'] < irank3_flag['user-item_dup_cnt_MAX'])
    irank3_flag = irank3_flag.drop([col for col in irank3_flag.columns if 'user-item' in col], 1)
    feature = feature.merge(irank2_flag, how='left', on=['buyer_admin_id', 'item_id'])
    feature = feature.merge(irank3_flag, how='left', on=['buyer_admin_id', 'item_id'])
    feature = feature.merge(dup_feature, how='left', on=['buyer_admin_id', 'item_id'])
    feature.to_hdf('../feature/rank/user_item_dupli_feature', 'all')
    print('>>> get_user_item_dupli_feature success')
    return feature


# In[6]:


def get_user_feature(df):
    """
    用户基础统计特征：
    
    1. 行为数： #TODO: 用户下单数 * 划窗
    2. 行为时间：去重数，首次，末次，首末差
        i: 天
        ii: 小时
        iii: 秒 
    3. 品类数：去重
    4. 店铺数：去重
    5. 商品数：去重
    6. 商品价格：最大、最小、平均
    
    备注：线下：0.8795→0.8795  提升：0
    备注：线下：0.8854→0.8850  提升：-0.0004
    ---------------------------------------------
    
    """
    feature_type = {
        'buyer_admin_id' : ['count'],
        'day':['nunique', 'max', 'min', np.ptp],
        'second':['max', 'min', np.ptp],
        'item_id':['nunique'],
        'cate_id':['nunique'],
        'store_id':['nunique'],
        'item_price': ['max', 'min', 'mean'],
    }
    df = filter_sample(df)
    feature = group_func(df, feature_type, group_key=['buyer_admin_id'])
    feature = add_prefix(feature, ['buyer_admin_id'], 'user_')
    feature.to_hdf('../feature/rank/user_feature', 'all')
    print('>>> user feature success')
    return feature


# In[7]:


def get_cate_feature(df):
    """
    品类基础特征：
    1. 行为数
    2. 行为时间：去重，首次，末次，首末差
        i: 天
        iii: 秒
    3. 用户数：去重
    4. 商品数：去重
    5. 店铺数：去重
    6. 商品价格：最大、最小、差值
    
    备注：线下：0.8795→0.8795  提升：0
    ---------------------------------------------
    """
    feature_type = {
        'cate_id' : ['count'],
        'buyer_admin_id' : ['nunique'],
        'item_id' :['nunique'],
        'store_id' : ['nunique'],
        'item_price': ['min', 'max', np.ptp],
        'day': ['max', 'min', 'nunique'],
        'second' : ['max', 'min', 'nunique', np.ptp],
    }

    feature = group_func(df, feature_type, group_key=['cate_id'])
    feature = add_prefix(feature, ['cate_id'], 'cate_')
    feature.to_hdf('../feature/rank/cate_feature', 'all')
    print('>>> cate feature success')
    return feature


# In[8]:


def get_store_feature(df):
    """
    店铺基础特征：
    1. 行为数： #TODO: 用户下单数 * 划窗
    2. 行为时间：去重，首次，末次，首末差
        i: 天
        iii: 秒
    3. 用户数：去重
    4. 商品数：去重
    5. 品类数：去重
    6. 商品价格：最大、最小、差值
    
    备注：线下：0.8795→0.8795  提升：0
    ---------------------------------------------
    
    """
    feature_type = {
        'store_id' : ['count'],
        'buyer_admin_id' : ['nunique'],
        'item_id' :['nunique'],
        'cate_id' : ['nunique'],
        'item_price': ['min', 'max', np.ptp],
        'day': ['max', 'min', 'nunique'],
        'second' : ['max', 'min', 'nunique', np.ptp],
    }

    feature = group_func(df, feature_type, group_key=['store_id'])
    feature = add_prefix(feature, ['store_id'], 'store_')
    feature.to_hdf('../feature/rank/store_feature', 'all')
    print('>>> store feature success')
    return feature


# In[9]:


def get_item_feature(df):
    """
    商品基础特征：
    1. 行为数： #TODO: 用户下单数 * 划窗
    2. 行为时间：去重，首次，末次，首末差
        i: 天
        ii: 小时
        iii: 秒
    3. 品类数：去重
    6. 商品价格：最大、最小、平均、求和、var、std
    3. 用户数：去重
    
    备注：线下：0.8795→0.8795  提升：0
    ---------------------------------------------
    """
    feature_type = {
        'item_id' : ['count'],
        'buyer_admin_id' : ['nunique'],
        'day': ['max', 'min', 'nunique'],
        'second' : ['max', 'min', 'nunique', np.ptp],
    }
    feature = group_func(df, feature_type, group_key=['item_id'])
    feature = add_prefix(feature, ['item_id'], 'item_')
    feature.to_hdf('../feature/rank/item_feature', 'all')
    print('>>> item feature success')
    return feature


# In[11]:


def get_user_cate_feature(df, name='all'):
    """
    商品 * 品类基础特征：
    1. 行为数： #TODO: 用户下单数 * 划窗
    2. 行为时间：去重，首次，末次，首末差
        i: 天
        ii: 小时
        iii: 秒
    3. 店铺数：去重
    4. 商品数：去重
    5. 商品价格：最大、最小、平均、求和、var、std
    6. 用户数：去重
    
    备注：线下：0.8697→0.8764  提升：0.0067
    ---------------------------------------------
    
    """
    feature_type = {
        'item_id' : ['nunique'],
        'second' : ['nunique', 'max', 'min', 'mean', 'std', np.ptp],
        'first_second_diff':['max', 'min', 'mean'],
        'last_second_diff':['max', 'min', 'mean'], 
        'day':['nunique', 'max', 'min', np.ptp],
        'item_price': ['max', 'min'],
        'dense_rank': ['max', 'min', 'mean', 'std', np.ptp],
    }
    
    df = filter_sample(df, 'cate_id')
    feature = group_func(df, feature_type, group_key=['buyer_admin_id', 'cate_id'])
    feature = add_prefix(feature, ['buyer_admin_id', 'cate_id'], 'user_cate_' + name + '_')
    feature.to_hdf('../feature/rank/user_cate_feature', name)
    print('>>> user_cate feature success')
    return feature


# In[12]:


def get_user_store_feature(df, name='all'):
    """
    商品 * 品类基础特征：
    1. 行为数： #TODO: 用户下单数 * 划窗
    2. 行为时间：去重，首次，末次，首末差
        i: 天
        ii: 小时
        iii: 秒
    3. 店铺数：去重
    4. 商品数：去重
    5. 商品价格：最大、最小、平均、求和、var、std
    6. 用户数：去重
    
    备注：线下：0.8764→0.8773  提升：0.001
    ---------------------------------------------
    sample = get_sample(dtype='rank')
    df = get_hdf('buy', if_filter_label=True, if_drop_duplicates=True)
    get_user_cate_feature(df, 'buy')
    get_user_store_feature(df, 'buy')
    """
    feature_type = {
        'item_id' : ['nunique'],
        'cate_id' : ['nunique'],
        'second' : ['nunique', 'max', 'min', 'mean', 'std', np.ptp],
        'first_second_diff':['max', 'min', 'mean'],
        'last_second_diff':['max', 'min', 'mean'],
        'day':['nunique', 'max', 'min', np.ptp],
        'item_price': ['max', 'min'],
        'dense_rank': ['max', 'min', 'mean', 'std', np.ptp],
    }
    
    df = filter_sample(df, 'store_id')
    feature = group_func(df, feature_type, group_key=['buyer_admin_id', 'store_id'])
    feature = add_prefix(feature, ['buyer_admin_id', 'store_id'], 'user_store_' + name + '_')
    feature.to_hdf('../feature/rank/user_store_feature', name)
    print('>>> user_store feature success')
    return feature


# In[13]:


def get_user_item_feature(df):
    """
    用户 * 商品基础特征：
    
    1. 行为数：计数
    2. 行为时间：去重，首次，末次，首末差
        i: 天
        ii: 小时
        iii: 秒
    3. 商品价格：求和
    
    备注：线下：0→0.88696  提升:
    ---------------------------------------------
    
    """
    feature_type = {
        'item_id':['count'],
        'day':['nunique', 'max', 'min', 'mean', np.ptp],
        'first_second_diff':['max', 'min', 'mean'], 
        'last_second_diff':['max', 'min', 'mean'], 
        'second':['nunique', 'max', 'min', 'mean', np.ptp],
        'irank':['max', 'min', 'mean', 'std', np.ptp],
    }
    
    df = filter_sample(df, 'item_id')
    feature = group_func(df, feature_type, group_key=['buyer_admin_id', 'item_id'])
    feature = add_prefix(feature, ['buyer_admin_id', 'item_id'], 'user_item_')
    feature.to_hdf('../feature/rank/user_item_feature', 'all')
    print('>>> user_item feature success')
    return feature


# In[14]:


def get_user_item_dedup_feature(df):
    """
    用户 * 商品基础特征：
    
    1. 行为数：计数
    2. 行为时间：去重，首次，末次，首末差
        i: 天
        ii: 小时
        iii: 秒
    3. 商品价格：求和
    
    备注：线下：0→0.88696  提升:
    ---------------------------------------------
    
    """
    feature_type = {
        'item_id':['count'],
        'day':['nunique', 'max', 'min', 'mean', np.ptp],
        'first_second_diff':['max', 'min', 'mean'], 
        'last_second_diff':['max', 'min', 'mean'], 
        'second':['nunique', 'max', 'min', 'mean', np.ptp],
        'dense_rank': ['max', 'min', 'mean', 'std', np.ptp],
    }
    
    df = filter_sample(df, 'item_id')
    feature = group_func(df, feature_type, group_key=['buyer_admin_id', 'item_id'])
    feature = add_prefix(feature, ['buyer_admin_id', 'item_id'], 'user_item_dedup_')
    feature.to_hdf('../feature/rank/user_item_dedup_feature', 'all')
    print('>>> user_item_dedup_feature feature success')
    return feature


# In[15]:


def get_user_cate_dedup_feature(df):
    """
    商品 * 品类基础特征：
    1. 行为数： #TODO: 用户下单数 * 划窗
    2. 行为时间：去重，首次，末次，首末差
        i: 天
        ii: 小时
        iii: 秒
    3. 店铺数：去重
    4. 商品数：去重
    5. 商品价格：最大、最小、平均、求和、var、std
    6. 用户数：去重
    
    备注：线下：0.8697→0.8764  提升：0.0067
    ---------------------------------------------
    
    """
    feature_type = {
        'item_id' : ['nunique'],
        'store_id' : ['nunique'],
        'second' : ['nunique', 'max', 'min', 'mean', 'std', np.ptp],
        'first_second_diff':['max', 'min', 'mean'],
        'last_second_diff':['max', 'min', 'mean'], 
        'day':['nunique', 'max', 'min', np.ptp],
        'item_price': ['max', 'min'],
        'dense_rank':['max', 'min', 'mean', 'std', np.ptp],
    }
    df = filter_sample(df, 'cate_id')
    feature = group_func(df, feature_type, group_key=['buyer_admin_id', 'cate_id'])
    feature = add_prefix(feature, ['buyer_admin_id', 'cate_id'], 'user_cate_dedup_')
    feature.to_hdf('../feature/rank/user_cate_dedup_feature', 'all')
    print('>>> user_cate_dedup_feature feature success')
    return feature


# In[16]:


def get_user_store_dedup_feature(df):
    """
    商品 * 品类基础特征：
    1. 行为数： #TODO: 用户下单数 * 划窗
    2. 行为时间：去重，首次，末次，首末差
        i: 天
        ii: 小时
        iii: 秒
    3. 店铺数：去重
    4. 商品数：去重
    5. 商品价格：最大、最小、平均、求和、var、std
    6. 用户数：去重
    
    备注：线下：0.8697→0.8764  提升：0.0067
    ---------------------------------------------
    
    """
    feature_type = {
        'item_id' : ['nunique'],
        'cate_id' : ['nunique'],
        'second' : ['nunique', 'max', 'min', 'mean', 'std', np.ptp],
        'first_second_diff':['max', 'min', 'mean'],
        'last_second_diff':['max', 'min', 'mean'],
        'day':['nunique', 'max', 'min', np.ptp],
        'item_price': ['max', 'min'],
        'irank_dedup':['max', 'min', 'mean', 'std', np.ptp],
    }
    df = filter_sample(df, 'store_id')
    feature = group_func(df, feature_type, group_key=['buyer_admin_id', 'store_id'])
    feature = add_prefix(feature, ['buyer_admin_id', 'store_id'], 'user_store_dedup_')
    feature.to_hdf('../feature/rank/user_store_dedup_feature', 'all')
    print('>>> user_store_dedup_feature success')
    return feature


# In[17]:


def get_user_item_lastday_dedup_feature(df):
    """
    在用户最后行为当天内，用户 * 商品基础特征：
    
    1. 行为数：计数
    2. 行为时间：去重，首次，末次，首末差
        i: 天
        ii: 小时
        iii: 秒
    3. 商品价格：求和
    
    备注：线下：0.8795→0.8783  提升:-0.0012
    ---------------------------------------------
    """
    
    feature_type = {
        'item_id':['count'],
        'day':['nunique', 'max', 'min', 'mean', np.ptp],
        'first_second_diff':['max', 'min', 'mean'], 
        'last_second_diff':['max', 'min', 'mean'], 
        'second':['nunique', 'max', 'min', 'mean', np.ptp],
        'dense_rank': ['max', 'min', 'mean', 'std', np.ptp],
    }
    df = filter_sample(df, 'item_id')
    feature = group_func(df, feature_type, group_key=['buyer_admin_id', 'item_id'])
    feature = add_prefix(feature, ['buyer_admin_id', 'item_id'], 'user_item_lastday_dedup_')
    feature.to_hdf('../feature/rank/user_item_lastday_dedup_feature', 'all')
    print('>>> user_item_lastday_dedup_feature feature success')
    return feature


# In[18]:


def get_user_cate_lastday_dedup_feature(df):
    """
    在用户最后行为当天内，用户 * 商品基础特征：
    
    1. 行为数：计数
    2. 行为时间：去重，首次，末次，首末差
        i: 天
        ii: 小时
        iii: 秒
    3. 商品价格：求和
    
    备注：线下：0.8795→0.8783  提升:-0.0012
    ---------------------------------------------
    last_day = df.groupby(['buyer_admin_id'])['day'].max().to_frame('last_day').reset_index()
    df = df.merge(last_day, on=['buyer_admin_id'], how='left')
    df = df[df['day']==df['last_day']]
    """
    
    feature_type = {
        'item_id' : ['nunique'],
        'store_id' : ['nunique'],
        'second' : ['nunique', 'max', 'min', 'mean', 'std', np.ptp],
        'first_second_diff':['max', 'min', 'mean'],
        'last_second_diff':['max', 'min', 'mean'], 
        'day':['nunique', 'max', 'min', np.ptp],
        'item_price': ['max', 'min'],
        'dense_rank':['max', 'min', 'mean', 'std', np.ptp],
    }
    df = filter_sample(df, 'cate_id')
    feature = group_func(df, feature_type, group_key=['buyer_admin_id', 'cate_id'])
    feature = add_prefix(feature, ['buyer_admin_id', 'cate_id'], 'user_cate_lastday_dedup_')
    feature.to_hdf('../feature/rank/user_cate_lastday_dedup_feature', 'all')
    print('>>> user_cate_lastday_dedup_feature success')
    return feature


# In[19]:


def get_user_store_lastday_dedup_feature(df):
    """
    在用户最后行为当天内，用户 * 商品基础特征：
    
    1. 行为数：计数
    2. 行为时间：去重，首次，末次，首末差
        i: 天
        ii: 小时
        iii: 秒
    3. 商品价格：求和
    
    备注：线下：0.8795→0.8783  提升:-0.0012
    ---------------------------------------------
    """
    
    feature_type = {
        'item_id' : ['nunique'],
        'cate_id' : ['nunique'],
        'first_second_diff':['max', 'min', 'mean'], 
        'last_second_diff':['max', 'min', 'mean'], 
        'second':['max', 'min', 'mean', np.ptp],
        'item_price': ['max', 'min', 'mean', 'sum', 'std'],
        'dense_rank':['max', 'min', 'mean', 'std'],
    }
    df = filter_sample(df, 'store_id')
    feature = group_func(df, feature_type, group_key=['buyer_admin_id', 'store_id'])
    feature = add_prefix(feature, ['buyer_admin_id', 'store_id'], 'user_store_lastday_dedup_')
    feature.to_hdf('../feature/rank/user_store_lastday_dedup_feature', 'all')
    print('>>> user_store_lastday_dedup_feature success')
    return feature


# In[33]:


def get_user_rank_feature(feature, feature_name, key, group_key=[], ascending=False):
    """
    import pandas
    name = 'user_item_dedup_feature'
    feat = pandas.read_hdf('../feature/rank/' + name, 'all')
    item = pandas.read_csv('../data/Antai_AE_round2_item_attr_20190813.zip')[['item_id', 'cate_id', 'store_id']]
    feature = pandas.merge(feat, item, how='left', on=['item_id'])

    get_user_rank_feature(feature, name, key=['item_id'], group_key=[] ,ascending=True)
    get_user_rank_feature(feature, name, key=['item_id'], group_key=[], ascending=False)
    get_user_rank_feature(feature, name, key=['item_id'], group_key=['cate_id'] ,ascending=True)
    get_user_rank_feature(feature, name, key=['item_id'], group_key=['cate_id'], ascending=False)
    get_user_rank_feature(feature, name, key=['item_id'], group_key=['store_id'] ,ascending=True)
    get_user_rank_feature(feature, name, key=['item_id'], group_key=['store_id'], ascending=False)

    name = 'user_cate_dedup_feature'
    feature = pandas.read_hdf('../feature/rank/' + name, 'all')
    get_user_rank_feature(feature, name, key=['cate_id'], group_key=[] ,ascending=True)
    get_user_rank_feature(feature, name, key=['cate_id'], group_key=[], ascending=False)

    name = 'user_store_dedup_feature'
    feature = pandas.read_hdf('../feature/rank/' + name, 'all')
    get_user_rank_feature(feature, name, key=['store_id'], group_key=[] ,ascending=True)
    get_user_rank_feature(feature, name, key=['store_id'], group_key=[], ascending=False)
    
    name = 'user_second_diff_feature'
    feature = pandas.read_hdf('../feature/rank/' + name, 'user')
    get_user_rank_feature(feature, name, key=[], group_key=[] ,ascending=True)
    get_user_rank_feature(feature, name, key=[], group_key=[], ascending=False)    

    name = 'user_second_diff_feature'
    feature = pandas.read_hdf('../feature/rank/' + name, 'item_id')
    get_user_rank_feature(feature, name, key=[], group_key=[] ,ascending=True)
    get_user_rank_feature(feature, name, key=[], group_key=[], ascending=False)
    
    name = 'user_second_diff_feature'
    feature = pd.read_hdf('../feature/rank/' + name, 'item_id')
    item = pd.read_csv('../data/Antai_AE_round2_item_attr_20190813.zip')[['item_id', 'cate_id', 'store_id']]
    feature = pd.merge(feature, item, how='left', on=['item_id'])

    import modin.pandas
    get_user_rank_feature(feature.copy(), name, key=['item_id'], group_key=[] ,ascending=True)
    get_user_rank_feature(feature.copy(), name, key=['item_id'], group_key=[], ascending=False)
    get_user_rank_feature(feature.copy(), name, key=['item_id'], group_key=['cate_id'] ,ascending=True)
    get_user_rank_feature(feature.copy(), name, key=['item_id'], group_key=['cate_id'], ascending=False)
    get_user_rank_feature(feature.copy(), name, key=['item_id'], group_key=['store_id'] ,ascending=True)
    get_user_rank_feature(feature.copy(), name, key=['item_id'], group_key=['store_id'], ascending=False)
    
    name = 'user_item_lastday_dedup_feature'
    feature = pd.read_hdf('../feature/rank/' + name, 'all')
    item = pd.read_csv('../data/Antai_AE_round2_item_attr_20190813.zip')[['item_id', 'cate_id', 'store_id']]
    feature = pd.merge(feature, item, how='left', on=['item_id'])

    import modin.pandas
    get_user_rank_feature(feature.copy(), name, key=['item_id'], group_key=[] ,ascending=True)
    get_user_rank_feature(feature.copy(), name, key=['item_id'], group_key=[], ascending=False)
    get_user_rank_feature(feature.copy(), name, key=['item_id'], group_key=['cate_id'] ,ascending=True)
    get_user_rank_feature(feature.copy(), name, key=['item_id'], group_key=['cate_id'], ascending=False)
    get_user_rank_feature(feature.copy(), name, key=['item_id'], group_key=['store_id'] ,ascending=True)
    get_user_rank_feature(feature.copy(), name, key=['item_id'], group_key=['store_id'], ascending=False)
    ------------------------------------------------------------------------------------------
    'user_item_dedup_feature', 'asc'
    'user_item_dedup_feature', 'desc'
    'user_item_dedup_feature', 'cate_id_asc'
    'user_item_dedup_feature', 'cate_id_desc
    'user_item_dedup_feature', 'store_id_asc'
    'user_item_dedup_feature', 'store_id_desc
    
    'user_cate_dedup_feature', 'asc'
    'user_cate_dedup_feature', 'desc'    

    'user_store_dedup_feature', 'asc'
    'user_store_dedup_feature', 'desc'
    
    'user_second_diff_feature', 'asc'
    'user_second_diff_feature', 'desc'
    
    'user_second_diff_feature', 'cate_id_asc'
    'user_second_diff_feature', 'cate_id_desc'
    
    'user_second_diff_feature', 'store_id_asc'
    'user_second_diff_feature', 'store_id_desc'
    
    'user_item_lastday_dedup_feature', 'asc'
    'user_item_lastday_dedup_feature', 'desc'
    
    'user_item_lastday_dedup_feature', 'cate_id_asc'
    'user_item_lastday_dedup_feature', 'cate_id_desc'
    
    'user_item_lastday_dedup_feature', 'store_id_asc'
    'user_item_lastday_dedup_feature', 'store_id_desc'
    
    用户 * 商品 * 排序 基础特征：
    
    备注：
    1. desc降序：线下：0.8795→0.8795  提升:0
    1. asc升序：线下：0.8795→0.8813  提升:-0.0018
    ---------------------------------------------
    """
    if ascending:
        name = 'asc'
    else:
        name = 'desc'
    columns = []
    for col in feature.columns:
        if col not in ['buyer_admin_id', 'item_id', 'cate_id', 'store_id']:
            column_name = col + '_rank_' + name
            feature[column_name] = feature.groupby(['buyer_admin_id'] + group_key)[col].rank(ascending=ascending, method='dense')
            columns.append(column_name)
            
    if len(group_key)>0:
        feature = feature[['buyer_admin_id', 'item_id'] + group_key + columns]
        name = group_key[0] + '_' + name
    else:
        feature = feature[['buyer_admin_id']+ key + columns]
        
    feature.to_hdf('../feature/rank/' + feature_name, name)
    print('>>> user_rank_feature feature success')
    return feature


# In[15]:


def get_user_second_diff_feature(df):
    """
    df = get_hdf('all', if_filter_label=True, if_drop_duplicates=True)
    import modin.pandas
    get_user_item_dupli_feature(df.copy())
    用户时间间隔统计特征：
    聚合层级：cate_id, store_id, item_id
    
    1. 商品与下个商品间隔
    2. 商品与下个同样商品间隔
    3. 商品与下个同品类商品间隔
    4. 商品与下个同店铺商品间隔
    
    备注：线下：0.8843→0.8852  提升：0.009
    ---------------------------------------------
    
    """
    df = df[['buyer_admin_id', 'store_id', 'cate_id', 'item_id', 'second']].drop_duplicates()
    df['second_diff'] = df['second'] - df.groupby(['buyer_admin_id'])['second'].shift(1)
    df['cate_id_second_diff'] = df['second'] - df.groupby(['buyer_admin_id', 'cate_id'])['second'].shift(1)
    df['store_id_second_diff'] = df['second'] - df.groupby(['buyer_admin_id', 'store_id'])['second'].shift(1)
    
    feature_type = {
        'second_diff' : ['max', 'min', 'mean', 'std', np.ptp],
        'cate_id_second_diff':['max', 'min', 'mean', 'std', np.ptp],
        'store_id_second_diff':['max', 'min', 'mean', 'std', np.ptp],
    }
    
    df = filter_sample(df)
    feature = group_func(df, feature_type, group_key=['buyer_admin_id'])
    feature = add_prefix(feature, ['buyer_admin_id'], 'user_second_diff_')
    feature.to_hdf('../feature/recall/user_second_diff_feature', 'user')
    
    for level in ['item_id', 'cate_id', 'store_id']:
        feature = group_func(df, feature_type, group_key=['buyer_admin_id', level])
        feature = add_prefix(feature, ['buyer_admin_id', level], 'user_' + level + '_second_diff_')
        feature.to_hdf('../feature/recall/user_second_diff_feature', level)
    print('>>> user_second_diff_feature success')
    return feature


# In[38]:


def get_user_prop_feature(df):
    """
    备注：
    线下：0.8795→0.8786  提升:-0.009
    线下：0.8799→0.8800  提升:+0.001
    ---------------------------------------------
    """
    df = df.drop_duplicates(subset=['buyer_admin_id', 'item_id', 'second'], keep='first')
    df = filter_sample(df)
    
    feature = df.groupby(['buyer_admin_id'])['item_id'].value_counts(normalize=True).to_frame('item_id_prop').reset_index()
    feature.to_hdf('../feature/rank/user_prop_feature', 'item_id_prop')
    
    feature = df.groupby(['buyer_admin_id'])['cate_id'].value_counts(normalize=True).to_frame('cate_id_prop').reset_index()
    feature.to_hdf('../feature/rank/user_prop_feature', 'cate_id_prop')
    
    feature = df.groupby(['buyer_admin_id'])['store_id'].value_counts(normalize=True).to_frame('store_id_prop').reset_index()
    feature.to_hdf('../feature/rank/user_prop_feature', 'store_id_prop')
    
    df = filter_sample(df, 'cate_id')
    feature = df.groupby(['buyer_admin_id', 'cate_id'])['item_id'].value_counts(normalize=True).to_frame('item_id_prop_cate_id').reset_index()
    feature.to_hdf('../feature/rank/user_prop_feature', 'item_id_prop_cate_id')
    
    feature = df.groupby(['buyer_admin_id', 'cate_id'])['store_id'].value_counts(normalize=True).to_frame('store_id_prop_cate_id').reset_index()
    feature.to_hdf('../feature/rank/user_prop_feature', 'store_id_prop_cate_id')
    
    df = filter_sample(df, 'store_id')
    feature = df.groupby(['buyer_admin_id', 'store_id'])['item_id'].value_counts(normalize=True).to_frame('item_id_prop_store_id').reset_index()
    feature.to_hdf('../feature/rank/user_prop_feature', 'item_id_prop_store_id')
    
    feature = df.groupby(['buyer_admin_id', 'store_id'])['cate_id'].value_counts(normalize=True).to_frame('cate_id_prop_store_id').reset_index()
    feature.to_hdf('../feature/rank/user_prop_feature', 'cate_id_prop_store_id')


# In[39]:


def get_user_item_irank_feature(df):
    """
    用户 * irank2商品基础特征：
    
    1. 是否与irank2相同item_id
    2. 是否与irank2相同cate_id
    3. 是否与irank2相同store_id
    """
    feature = df[['buyer_admin_id', 'item_id', 'cate_id', 'store_id']].drop_duplicates()
    
    for i in range(2, 6):
        irank = df[df['irank']==i][['buyer_admin_id', 'item_id', 'cate_id', 'store_id']]
        irank.columns = ['buyer_admin_id', f'irank{i}_item_id', f'irank{i}_cate_id', f'irank{i}_store_id']
        feature = pd.merge(feature, irank, how='left', on='buyer_admin_id')

    for col in feature.columns:
        if 'irank' in col:
            feature[col] = (feature[col]==feature[col.split('_', 1)[1]]).astype(int)
    
    feature = feature.drop(['cate_id', 'store_id'], 1)
    feature.to_hdf('../feature/rank/user_item_irank_feature', 'all')
    print('>>> user-item-irank feature success')
    return feature


# In[9]:


def get_user_rank_diff_feature(df, groupby_key=[]):
    """
    TODO: 用户下rank
    用户商品间隔特征
        1. 商品最后一次浏览过后，还浏览了多少同品类、同店铺的商品，按次数倒序 key-first
        
    
    """
    
    df['item_rank_diff'] = df['dense_rank'] - df.groupby(['buyer_admin_id', 'item_id'])['dense_rank'].shift(1)
    
    for col in ['cate_id', 'store_id']:
        df[col + '_rank'] = df.groupby(['buyer_admin_id', col]).cumcount()
        df[col + '_rank_diff'] = df[col + '_rank'] - df.groupby(['buyer_admin_id', 'item_id'])[col + '_rank'].shift(1)
#         df.loc[df[col + '_rank_diff'].isnull(), col + '_rank_diff'] = df[col + '_rank']
    
#     df['cate_id_rank_diff'] = df['dense_rank'] - df.groupby(['buyer_admin_id', 'cate_id'])['dense_rank'].shift(1)
#     df['store_id_rank_diff'] = df['dense_rank'] - df.groupby(['buyer_admin_id', 'store_id'])['dense_rank'].shift(1)
    
    feature_type = {
        'store_id_rank' : ['first', 'last'],
        'cate_id_rank' : ['first', 'last'], # rank 0.0002
        'cate_id_rank_diff' : ['first', 'max', 'min', 'mean'],
        'store_id_rank_diff' : ['first', 'max', 'min', 'mean'],
        'item_rank_diff': ['first', 'max', 'min', 'mean'],
    }
    
    df = filter_sample(df)
    feature = group_func(df, feature_type, group_key=['buyer_admin_id', 'item_id'])
    
    for col in feature.columns:
        if col not in ['buyer_admin_id', 'item_id', 'cate_id', 'store_id']:
            if len(groupby_key) > 0:
                name = col + '_'+ groupby_key[0] + '_rank_asc'
            else:
                name = col + '_rank_asc'
            feature[name] = feature.groupby(['buyer_admin_id'] + groupby_key)[col].rank(ascending=True, method='dense')  

    feature = add_prefix(feature, ['buyer_admin_id', 'item_id', 'cate_id', 'store_id'], 'user_rank_diff')
    feature.to_hdf('../feature/rank/user_rank_diff_feature', 'all')
    print('>>> user_item_rank_diff_feature feature success')
    return feature


# In[41]:


def get_user_item_rank_diff_rank_cate(feature, ascending=False):
    """
    用户 * 商品 * 排序(cate_id) 基础特征：
    
    备注：
    1. desc降序：线下：0.8880→0.8884  提升:+0.0004
    1.  asc升序：线下：0.8880→0.8885  提升:+0.0005
    ---------------------------------------------
    """
    if ascending:
        name = 'asc'
    else:
        name = 'desc'
    columns = []
    if 'cate_id' not in feature.columns:
        item = pd.read_csv('../data/Antai_AE_round2_item_attr_20190813.zip')[['item_id', 'cate_id']]
        feature = pd.merge(feature, item, how='left', on=['item_id'])
    
    for col in feature.drop(['buyer_admin_id', 'item_id', 'cate_id'], 1).columns:
        column_name = col + '_RANK_' + 'name'
        feature[column_name] = feature.groupby(['buyer_admin_id', 'cate_id'])[col].rank(ascending=ascending, method='dense')
        columns.append(column_name)
        
    feature = feature[['buyer_admin_id', 'item_id', 'cate_id'] + columns]
    feature.to_hdf('../feature/rank/user_item_rank_diff_rank_cate', name)
    print('>>> user_item_rank_diff_rank_cate feature success')
    return feature


# In[8]:


def get_item_conv_feature(df):
    """
    商品转化率特征
    df = get_hdf('all', if_filter_label=True, if_drop_duplicates=True)
    get_item_conv_feature(df)
    """
    item_pv = df.drop_duplicates(subset=['buyer_admin_id', 'item_id', 'second']).groupby(['item_id']).size().to_frame('pv').reset_index()
    item_uv = df.groupby(['item_id'])['buyer_admin_id'].nunique().to_frame('uv').reset_index()
    item_buy_uv = df[df['buy_flag']==1].groupby(['item_id'])['buyer_admin_id'].nunique().to_frame('buy_uv').reset_index()

    dup = df[df['buy_flag']==1][df.duplicated(subset=['buyer_admin_id', 'item_id', 'second'], keep=False)]
    multi_buy_uv = dup.groupby(['item_id'])['buyer_admin_id'].nunique().to_frame('multi_buy_uv').reset_index()

    view_time = df.groupby(['buyer_admin_id', 'item_id']).size().to_frame('user_view_time').reset_index()
    view_one_time = view_time.groupby(['item_id'])['user_view_time'].value_counts(normalize=True).to_frame('view_onetime_prop').reset_index()
    view_one_time = view_one_time[view_one_time['user_view_time']==1].drop(['user_view_time'],1 )
    
    last = df.drop_duplicates(subset=['buyer_admin_id'], keep='first')
    last_cnt = last.groupby(['item_id']).size().to_frame('last_buy').reset_index()
    
    last_via_day = df.drop_duplicates(subset=['buyer_admin_id', 'day'], keep='first')        .drop_duplicates(subset=['buyer_admin_id', 'item_id'], keep='first')
    last_via_day_cnt = last_via_day.groupby(['item_id']).size().to_frame('last_buy_day').reset_index()
    
    
    feature = item_pv.merge(item_uv, on=['item_id'], how='left')            .merge(item_buy_uv, on=['item_id'], how='left')            .merge(multi_buy_uv, on=['item_id'], how='left')            .merge(view_one_time, on=['item_id'], how='left')            .merge(last_cnt, on=['item_id'], how='left')            .merge(last_via_day_cnt, on=['item_id'], how='left').fillna(0)

    feature['pv/uv'] = feature['pv'] / feature['uv']
    feature['buy_uv/pv'] = feature['buy_uv'] / feature['uv']
    feature['multi_buy_uv/buy_uv'] = feature['multi_buy_uv'] / feature['buy_uv']
    feature['multi_buy_uv/uv'] = feature['multi_buy_uv'] / feature['uv']
    feature['last_buy/uv'] = feature['last_buy'] / feature['uv']
    feature['last_buy/buy_uv'] = feature['last_buy'] / feature['buy_uv']
    feature = feature.fillna(0)
    
    feature = add_prefix(feature, ['item_id'], 'item_conv_')
    feature.to_hdf('../feature/rank/item_conv_feature', 'all')
    print('>>> item_conv_feature success')
    return feature

