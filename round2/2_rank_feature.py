from round2.round2_base import get_sample, get_hdf

if __name__ == "__main__":
    sample = get_sample(dtype='rank')
    
    df = get_hdf('all', if_filter_label=True, if_drop_duplicates=True, if_lastday=True)
    print(df.head())
