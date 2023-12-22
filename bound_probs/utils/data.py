
import pandas as pd
import pkg_resources
import numpy as np

def get_dataset(ds='ds2'):

    if ds == 'ds2':
        stream = pkg_resources.resource_stream(__name__, '../data/ds2.csv')
        df = pd.read_csv(stream, index_col=[0, 1, 2])

        df['p'] = df['presentedProb_1'] / df['presentedProb_2']

        df['p_bin'] = pd.cut(df['p'], bins=np.arange(0, 1.05, .05), labels=np.arange(0.025, 1, 0.05))
        df['p_bin2'] = pd.cut(df['p'], bins=np.arange(0, 1.05, .1), labels=np.arange(0.05, 1.05, 0.1))
        df['p_bin3'] = pd.cut(df['p'], bins=np.arange(0, 1.1, .15), labels=np.arange(0.1, 1.1, 0.15))
        df['bias'] = df['certaintyEquivalent'] / 43. - df['p']

    else:
        raise NotImplementedError(f'Dataset {ds} not implemented')

    return df

