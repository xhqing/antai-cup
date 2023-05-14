import pandas as pd
import gc


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


if __name__ == "__main__":
    save_hdf()
