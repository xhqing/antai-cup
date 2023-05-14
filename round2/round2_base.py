#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import inspect
import os
import gc

# 查看hdf文件是否已经生成
def check_hdf(func, dtype='yyzz'):
    file = func.__name__.split('_', maxsplit=1)[1]
    path = f'../output/{file}'
    if os.path.exists(path):
        df = pd.read_hdf(path, dtype)
        print('>>> read ', path, dtype)
        return df
    else:
        print('>>> generate ', path, dtype)


def reduce_memory(data):
    start_memory = data.memory_usage().sum() / 1024**2 
    print("Memory usage of properties dataframe is :",start_memory," MB")
    NAlist = [] # Keeps track of columns that have missing values filled in. 
    
    for col in data.columns:
        if ('int' in data[col].dtype.name) or ('float' in data[col].dtype.name):  # Exclude strings
            try:
                # Print current column type
                print("******************************")
                print("Column: ",col)
                print("dtype before: ",data[col].dtype)

                # make variables for Int, max and min
                IsInt = False
                value_max = data[col].max()
                value_min = data[col].min()

                # Integer does not support NA, therefore, NA needs to be filled
                if not np.isfinite(data[col]).all(): 
                    NAlist.append(col)
                    data[col].fillna(value_min-1,inplace=True)  

                # test if column can be converted to an integer
                asint = data[col].fillna(0).astype(np.int64)
                result = (data[col] - asint)
                result = result.sum()
                if result > -0.01 and result < 0.01:
                    IsInt = True


                # Make Integer/unsigned Integer datatypes
                if IsInt:
                    if value_min >= 0:
                        if value_max < 255:
                            data[col] = data[col].astype(np.uint8)
                        elif value_max < 65535:
                            data[col] = data[col].astype(np.uint16)
                        elif value_max < 4294967295:
                            data[col] = data[col].astype(np.uint32)
                        else:
                            data[col] = data[col].astype(np.uint64)
                    else:
                        if value_min > np.iinfo(np.int8).min and value_max < np.iinfo(np.int8).max:
                            data[col] = data[col].astype(np.int8)
                        elif value_min > np.iinfo(np.int16).min and value_max < np.iinfo(np.int16).max:
                            data[col] = data[col].astype(np.int16)
                        elif value_min > np.iinfo(np.int32).min and value_max < np.iinfo(np.int32).max:
                            data[col] = data[col].astype(np.int32)
                        elif value_min > np.iinfo(np.int64).min and value_max < np.iinfo(np.int64).max:
                            data[col] = data[col].astype(np.int64)    

                # Make float datatypes 32 bit
                else:
                    data[col] = data[col].astype(np.float32)

                # Print new column type
                print("dtype after: ",data[col].dtype)
                print("******************************")
            except:
                print("dtype after: Failed")
        else:
            print("dtype remain: ",data[col].dtype)
    
    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    end_memory = data.memory_usage().sum() / 1024**2 
    print("Memory usage is: ",end_memory," MB")
    print("This is ",100*start_memory/end_memory,"% of the initial size")
    print("Missing Value list", NAlist)
    return data

# 检查提交文件是否正确
def check_subfile(df, name):
    label = pd.read_csv('../data/Antai_AE_round2_test_20190813.zip')
    df = df.merge(label[['buyer_admin_id']].drop_duplicates(), how='right', on=['buyer_admin_id']).fillna(0)
    
    if len(df) != 9844:
        raise ValueError('sub file lenght is not correct')
    
    if len(df.columns) != 31:
        raise ValueError('sub file column is not correct')
        
    if df.columns[0] != 'buyer_admin_id':
        raise ValueError('sub file columns dont contains buyer_country_id')
    df.to_csv('../submit/' + name, index=False, header=None)


# top30sku，行列转换
def submit_transform(df, name=None):
    # 过滤掉不在训练集的商品
    train_item = get_hdf('train')['item_id'].drop_duplicates()
    df = df[df['item_id'].isin(train_item)]
    df['irank'] = df.groupby(['buyer_admin_id']).cumcount()+1
    
    df = df[df['irank']<=30].set_index(['buyer_admin_id', 'irank']).unstack(-1)
    df = df.fillna(11717821).astype(int)
    df.columns = [int(col) if isinstance(col, float) else col for col in df.columns.droplevel(0)]
    df = df.reset_index()
    if name is None:
        name = datetime.datetime.today().strftime('%m-%d') + '.csv'
    check_subfile(df, name)
    return df

def save_hdf():
    train = pd.read_csv('./data/Antai_AE/Antai_AE_round2_train_20190813.csv')
    test = pd.read_csv('./data/Antai_AE/Antai_AE_round2_test_20190813.csv')
    item = pd.read_csv('./data/Antai_AE/Antai_AE_round2_item_attr_20190813.csv')

    df = pd.concat([train.assign(is_train=1), test.assign(is_train=0)])
    del train, test; gc.collect()
    df.sort_values(by=['country_id', 'buyer_admin_id', 'irank'], ascending=[1, 1, 0], inplace=True)
    
    # 创建时间信息列
    df['log_time'] = pd.to_datetime(df['log_time'])
    df['time_rank'] = df.groupby(['buyer_admin_id'])['log_time'].rank(ascending=False, method='dense')
   
    df['log_time'] = df['log_time'].astype(str)
    df['date'] = df['log_time'].str[:10]
    df['day'] = df['log_time'].str[8:10].astype(int)
    
    # 用户每条记录与首末(irank2为末)记录时间点
    # second = df[df['irank']>1].groupby(['buyer_admin_id']).agg(first_second_diff=('second','min'), last_second_diff=('second','max')).reset_index()
    # df = pd.merge(df, second, on=['buyer_admin_id'], how='left')
    # df['first_second_diff'] = df['second'] - df['first_second_diff']
    # df['last_second_diff'] = df['second'] - df['last_second_diff']

    # df.loc[df['irank']==1, 'first_second_diff'] = np.nan
    # df.loc[df['irank']==1, 'last_second_diff'] = np.nan
    
    df = df.sort_values(by=['country_id', 'buyer_admin_id', 'irank'], ascending=[1, 1, 1]).reset_index(drop=True)
    df = reduce_memory(df)
    df = pd.merge(df, item, how='left', on='item_id')
    memory = df.memory_usage().sum() / 1024**2 
    print('After memory usage of properties dataframe is :', memory, " MB")
    
    # 处理irank=1但是buy_flag为0的数据
    df = df.sort_values(by=['buyer_admin_id', 'log_time', 'buy_flag'], ascending=[1, 0, 0])
    df['irank'] = df.groupby(['buyer_admin_id']).cumcount() + 1
    df.loc[df['is_train']==0, 'irank'] = df.loc[df['is_train']==0, 'irank'] + 1
    # df['second_irankn'] = df['second']
    # df.loc[df['irank']==1, 'second_irankn'] = np.nan
    # df['dense_rank'] = df.groupby(['buyer_admin_id'])['second_irankn'].rank(method='dense', ascending=False)
    df = df.reset_index(drop=True)
    
    # 生成hdf文件
    path = "./data/base/"
    if not os.path.exists(path):
        os.makedirs(path)

    df[df['is_train']==1][:30000000].to_hdf('./data/base/train-half1.h5', key='df', mode="w")
    df[df['is_train']==1][30000000:].to_hdf('./data/base/train-half2.h5', key='df', mode="w")
    df[df['is_train']==0].to_hdf('./data/base/test.h5', key='df', mode="w")
    df[df['country_id']=='xx'].to_hdf('./data/base/test.h5', key='df', mode="w")
    df[df['country_id']=='yy'].to_hdf('./data/base/yy.h5', key='df', mode="w")
    df[df['country_id']=='zz'].to_hdf('./data/base/zz.h5', key='df', mode="w")
    df[df['buy_flag']==1].to_hdf('./data/base/buy.h5', key='df', mode="w")
    df[df['irank']==1].to_hdf('./data/base/irank1.h5', key='df', mode="w")
    df[df['irank']==2].to_hdf('./data/base/irank2.h5', key='df', mode="w")
    df[df['irank']==3].to_hdf('./data/base/irank3.h5', key='df', mode="w")
    df[df['irank']==4].to_hdf('./data/base/irank4.h5', key='df', mode="w")
    df[df['irank']==5].to_hdf('./data/base/irank5.h5', key='df', mode="w")
    return df

# 获取热销商品
def get_hot(dtype='all'):
    # 提取整体数据
    df = get_hdf(dtype='yyzz')
    
    # 每日热销商品与品类热销(top30)
    if dtype in ['date', 'cate_id', 'store_id']:
        # 计算某天/某品类下，top30热销商品
        hot = df.groupby([dtype, 'item_id'])['buyer_admin_id'].nunique().to_frame('count').reset_index().sort_values([dtype, 'count'],ascending=[1,0])
        hot = hot.groupby([dtype]).head(60).drop('count', 1)
        hot['irank'] = hot.groupby(dtype).cumcount() + 30
        hot.reset_index(drop=True).to_hdf('../data/hot.h5', dtype)
        return hot
    
    # 整体热销商品(top30)
    elif dtype=='all':
        buyer_admin_id = df['buyer_admin_id'].unique()
        item_id = df.groupby(['item_id'])['buyer_admin_id'].nunique().sort_values(ascending=False).head(60).index.tolist()
        hot = pd.DataFrame(index = pd.MultiIndex.from_product([buyer_admin_id, item_id], names=['buyer_admin_id', 'item_id'])).reset_index()
        hot['irank'] = hot.groupby(['buyer_admin_id']).cumcount() + 30
        hot.reset_index(drop=True).to_hdf('../data/hot.h5', 'all')
        return hot

def get_hdf(dtype='all', data_type='base', if_filter_label=False, if_lastday=False, if_drop_duplicates=False, if_debug=False, is_train=None):
    """
    data_type: base 原始数据文件 
    data_type: slide 向前滑一次购买记录
    data_type: slide_recall 冷启动划窗购买记录
    
        dtype='train'，训练集
        dtype='test'，测试集
        dtype='all'，训练集+测试集
        dtype='yyzz'，yy国和zz国数据
        dtype='buy'， buy_flag=1的所有数据 
        dtype='irank1'， irank=1的所有数据 
        dtype='irank2'， irank=2的所有数据 
    
    if_filter_label : 是否过滤label
    if_lastday ： 是否只取最后一天数据()
    if_drop_duplicates : 是否过滤重复数据(按秒去重)
    if_debug : debug取前10000行
    is_train : 是否只取训练数据
    """
    
    # 基础数据文件
    path = './data/' + data_type
    if dtype == 'all':
        df = pd.concat([pd.read_hdf(path+'/train-half1.h5'),
                        pd.read_hdf(path+'/train-half2.h5'),
                        pd.read_hdf(path+'/test.h5')])
    elif dtype == 'train':
        df = pd.concat([pd.read_hdf(path+'/train-half1.h5'),
                        pd.read_hdf(path+'/train-half2.h5')])
    elif dtype == 'yyzz':
        df = pd.concat([pd.read_hdf(path+'/yy.h5'),
                        pd.read_hdf(path+'/zz.h5')])
    else:
        df = pd.read_hdf(path, dtype)

    # 过滤irank=1，在做特征时使用，防止加入label信息
    if if_filter_label:
        df = df[df['irank']!=1]

    # 用户最后一天行为的数据, 做特征时，需要if_filter_label=True，防止把irank1的日期作为最后一天，泄露label信息
    if if_lastday:
        last_day = df.groupby(['buyer_admin_id'])['day'].max().to_frame('last_day').reset_index()
        df = df.merge(last_day, on=['buyer_admin_id'], how='left')
        df = df[df['day']==df['last_day']]

    # 过滤数据大量，同一数据重复的数据
    if if_drop_duplicates:
        df = df.drop_duplicates(subset=['buyer_admin_id', 'item_id', 'second'], keep='first').reset_index(drop=True)
        df['irank_dedup'] = df.groupby(['buyer_admin_id']).cumcount()
    
    if is_train is not None:
        if is_train:
            df = df[df['is_train']==1]
        else:
            df = df[df['is_train']==0]
            
    if if_debug:
        df = df[:100000]
    return df

def get_user(dtype, data_type='base'):
    """用户分群：
    all：全用户 624804
        train + test = all
        > train：训练集用户 614960
        
            按是否有历史购买记录划分：train_buy + non_train_buy = train 
            >> train_buy : 历史有购买记录用户（irank2 > 1 & buy_flag = 1）558565
            >> non_train_buy : 历史无购买记录用户（irank2 > 1 & buy_flag = 0） 56395
        
            按是否冷启动划分：rebuy_user + cold_user = train
            >> rebuy_user : irank1为历史复购 418821  --当前rank模型训练用户样本
            >> cold_user :  irank1为首次购买 196139  --当前recall模型训练用户样本
        
                cold_buy_user + non_cold_buy_user = cold_user
                >>> cold_buy_user : irank1为首次购买，但历史有购买记录 139746
                >>> non_cold_buy_user : irank1为首次购买，但是历史无购买记录 56393
        
        > test：测试集用户 9844
            按是否有历史购买记录划分：test_buy + non_test_buy = test 
            >> test_buy : 历史有购买记录用户（irank2 > 1 & buy_flag = 1）8947
            >> non_test_buy : 历史无购买记录用户（irank2 > 1 & buy_flag = 0）897
        
        recall : cold + test 205983
    """
    path = '../data/user_' + data_type
    if os.path.exists(path):
        user = pd.read_hdf(path, dtype).reset_index(drop=True)
        return user
    else:
        
        df = get_hdf(data_type=data_type)
        df['item_rank'] = df.groupby(['buyer_admin_id', 'item_id']).cumcount(ascending=False) + 1
        
        all_user = df[['buyer_admin_id', 'is_train', 'country_id']].drop_duplicates()
        all_user.to_hdf(path, 'all')

        train = all_user[all_user['is_train']==1]
        train.to_hdf(path, 'train')

        test = all_user[all_user['is_train']==0]
        test.to_hdf(path, 'test')

        xx = all_user[all_user['country_id']=='xx']
        xx.to_hdf(path, 'xx')

        yy = all_user[all_user['country_id']=='yy']
        yy.to_hdf(path, 'yy')

        zz= all_user[all_user['country_id']=='zz']
        zz.to_hdf(path, 'zz')

        yyzz = all_user[all_user['country_id']!='xx']
        yyzz.to_hdf(path, 'yyzz')
        
        buy = df[(df['irank']>1) & (df['buy_flag']==1)][['buyer_admin_id', 'is_train', 'country_id']].drop_duplicates()
        train_buy = buy[buy['is_train']==1]
        test_buy = buy[buy['is_train']==0]
        
        buy.to_hdf(path, 'buy')
        train_buy.to_hdf(path, 'train_buy')
        
        buy = df[(df['irank']>1) & (df['buy_flag']==1)][['buyer_admin_id', 'is_train', 'country_id']].drop_duplicates()
        train_buy = buy[buy['is_train']==1]
        non_train_buy = train[~train['buyer_admin_id'].isin(train_buy['buyer_admin_id'])]

        test_buy = buy[buy['is_train']==0]
        non_test_buy = test[~test['buyer_admin_id'].isin(test_buy['buyer_admin_id'])]

        buy.to_hdf(path, 'buy')
        train_buy.to_hdf(path, 'train_buy')
        non_train_buy.to_hdf(path, 'non_train_buy')
        test_buy.to_hdf(path, 'test_buy')
        non_test_buy.to_hdf(path, 'non_test_buy')
        
        rebuy = df[(df['irank']==1) & (df['item_rank']>1)][['buyer_admin_id', 'is_train', 'country_id']].drop_duplicates()
        rebuy.to_hdf(path, 'rebuy')

        cold = df[(df['irank']==1) & (df['item_rank']==1)][['buyer_admin_id', 'is_train', 'country_id']].drop_duplicates()
        cold.to_hdf(path, 'cold')

        cold_buy = cold[cold['buyer_admin_id'].isin(buy['buyer_admin_id'])]
        cold_buy.to_hdf(path, 'cold_buy')
        
        non_cold_buy = cold[~cold['buyer_admin_id'].isin(buy['buyer_admin_id'])]
        non_cold_buy.to_hdf(path, 'non_cold_buy')
        
        recall = pd.concat([cold ,test]).reset_index(drop=True)
        recall.to_hdf(path, 'recall')


def get_sample(dtype, data_type='base'):
    """样本选取
    all(34751160): 用户交互所有样本(排除irank1)，用于在冷启动预测中，过滤用户已历史交互的样本 
    rank(1410353): 用户buy_flag打标为1所有样本(buy_flag=1)，用于在rank模型中，irank=1为正样本, 其他为负样本
    baseline(1899886): 按训练集irank去重后升序排序 
    
    Tips:baseline包含训练集历史有购买用户，rank仅包含训练集复购用户与测试集全量用户
    """
    
    path = '../data/sample_' + data_type
    if os.path.exists(path):
        sample = pd.read_hdf(path, dtype).reset_index(drop=True)
        return sample
    else:
        df = get_hdf('all', data_type)
        rebuy_user = get_user('rebuy', data_type)
        test_user = get_user('test', data_type)
        cold_user = get_user('cold', data_type)

        all_sample = df[df['irank']!=1][['buyer_admin_id', 'item_id', 'country_id', 'cate_id', 'store_id']].drop_duplicates()
        all_sample.to_hdf(path, 'all')

        rank_user = pd.concat([rebuy_user ,test_user])
        rank_sample = df[(df['buy_flag']==1) & (df['buyer_admin_id'].isin(rank_user['buyer_admin_id']))]            [['buyer_admin_id', 'item_id', 'country_id', 'cate_id', 'store_id', 'is_train']].drop_duplicates()
        irank1 = get_hdf('irank1', data_type)[['buyer_admin_id', 'item_id']].assign(irank=1)
        rank_sample = rank_sample.merge(irank1, how='left', on=['buyer_admin_id', 'item_id'])
        rank_sample['irank'] = rank_sample['irank'].fillna(0)
        
        baseline = df[(df['irank']>1) & (df['is_train']==1) & (df['buy_flag']==1)][['buyer_admin_id', 'item_id']].drop_duplicates().reset_index(drop=True)
        baseline['irank'] = baseline.groupby(['buyer_admin_id']).cumcount() + 1
        baseline.to_hdf(path, 'baseline')
        
        rank_sample = pd.merge(rank_sample, baseline.rename(columns = {'irank' : 'baseline'}), on=['buyer_admin_id', 'item_id'], how='left')
        rank_sample.to_hdf(path, 'rank')

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

