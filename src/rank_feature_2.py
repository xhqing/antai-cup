from src.rank_feature import *
from round2_base import *


if __name__ == "__main__":
    sample = get_sample(dtype='rank')
    import pdb
    pdb.set_trace()
    df = get_hdf('all', if_filter_label=True, if_drop_duplicates=True, if_lastday=True)

