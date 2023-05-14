#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('run', 'round2_base.ipynb')


# sample = get_user('recall')
# df = get_hdf(dtype='all', if_filter_label=True, if_drop_duplicates=True)

# In[4]:


def add_prefix(df, exclude_columns, prefix):
    if isinstance(exclude_columns, str):
        exclude_columns = [exclude_columns]
        
    column_names = [col for col in df.columns if col not in exclude_columns]
    df.rename(columns = dict(zip(column_names, [prefix + name for name in column_names])), inplace=True)
    return df

def group_func(df, group_func_dic, group_key):
    if isinstance(group_func_dic, str):
        group_func_dic = [group_func_dic]
        
    features = df.groupby(group_key).agg(group_func_dic)
    features.columns = [e[0] + "_" + e[1].upper() for e in features.columns.tolist()]
    features.reset_index(inplace=True)
    return features

def filter_sample(df, key=None):
    if key is None:
        df = df.merge(sample[['buyer_admin_id']].drop_duplicates(), on=['buyer_admin_id'], how='inner')
    else:
        df = df.merge(sample[['buyer_admin_id', key]].drop_duplicates(), on=['buyer_admin_id', key], how='inner')
    return df


# In[5]:


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
        'dense_rank':['max', 'min', 'mean', 'std', np.ptp],
    }
    df = filter_sample(df)
    feature = group_func(df, feature_type, group_key=['buyer_admin_id', 'store_id'])
    feature = add_prefix(feature, ['buyer_admin_id', 'store_id'], 'user_store_dedup_')
    feature.to_hdf('../feature/recall/user_store_dedup_feature', 'all')
    print('>>> user_store_dedup_feature success')
    return feature


# In[6]:


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
        'second' : ['nunique', 'max', 'min', 'mean', np.ptp],
        'first_second_diff':['max', 'min', 'mean'],
        'last_second_diff':['max', 'min', 'mean'], 
        'day':['nunique', 'max', 'min', np.ptp],
        'item_price': ['max', 'min'],
        'dense_rank':['max', 'min', 'mean', np.ptp],
    }
    df = filter_sample(df)
    feature = group_func(df, feature_type, group_key=['buyer_admin_id', 'cate_id'])
    feature = add_prefix(feature, ['buyer_admin_id', 'cate_id'], 'user_cate_dedup_')
    feature.to_hdf('../feature/recall/user_cate_dedup_feature', 'all')
    print('>>> user_cate_dedup_feature feature success')
    return feature


# In[7]:


def get_item_feature(df, name='all'):
    """
    df = get_hdf(dtype='buy', if_filter_label=True, if_drop_duplicates=True)
    get_item_feature(df, name='buy')
    
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
    feature = add_prefix(feature, ['item_id'], 'item_' + name +'_')
    
    feature.to_hdf('../feature/recall/item_feature', name)
    print('>>> item feature success')
    return feature


# In[8]:


def get_cate_feature(df, name='all'):
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
    feature = add_prefix(feature, ['cate_id'], 'cate_' + name + '_')
    feature.to_hdf('../feature/recall/cate_feature', name)
    print('>>> cate feature success')
    return feature


# In[9]:


def get_store_feature(df, name='all'):
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
    feature = add_prefix(feature, ['store_id'], 'store_' + name + '_')
    feature.to_hdf('../feature/recall/store_feature', name)
    print('>>> store feature success')
    return feature


# In[10]:


def get_user_second_diff_feature(df):
    """
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
    
    for level in ['cate_id', 'store_id']:
        feature = group_func(df, feature_type, group_key=['buyer_admin_id', level])
        feature = add_prefix(feature, ['buyer_admin_id', level], 'user_' + level + '_second_diff_')
        feature.to_hdf('../feature/recall/user_second_diff_feature', level)
    print('>>> user_second_diff_feature success')
    return feature


# In[11]:


def get_item_conv_feature(df):
    """
    商品转化率特征
    
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
    feature.to_hdf('../feature/recall/item_conv_feature', 'all')
    print('>>> item_conv_feature success')
    return feature


# In[15]:


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
        
    feature.to_hdf('../feature/recall/' + feature_name, name)
    print('>>> user_rank_feature feature success')
    return feature

