import pandas as pd
import random
import numpy as np
df = pd.read_pickle('../data/pitt_cookie_dataframe_single_utterances.pickle')

class Ml_sets(object):
    def __init__(self,df,list_index):
        df_portion =  df.loc[df['id'].isin(list_index)]
        self.other_features = df_portion.loc[:, ~df_portion.columns.isin(['label', 'text','id'])].as_matrix()
        self.X = df_portion.text
        numeric_label = []
        for string in df_portion.label:
            if string == 'Dementia':
                numeric_label.append(1)
            if string == 'Control':
                numeric_label.append(0)
        self.y = np.array(numeric_label)

def train_test_split_same_patient(df,train_percentage,test_percentage,validation_percentage):
    if not train_percentage + test_percentage + validation_percentage ==1:
        raise AssertionError('Percentage for split are not valid')

    id_list = df.id.unique()
    random.shuffle(id_list)

    train_index = int(len(id_list)*train_percentage)
    validation_delta = int(len(id_list)*validation_percentage)

    train_portion = id_list[0:train_index]
    validation_portion = id_list[train_index+1:train_index+1+validation_delta]
    test_portion = id_list[train_index+validation_delta+2:len(id_list)-1]

    train_set = Ml_sets(df,train_portion)
    validation_set = Ml_sets(df,validation_portion)
    test_set = Ml_sets(df,test_portion)

    return train_set,validation_set,test_set

train_test_split_same_patient(df,0.8,0.1,0.1)
