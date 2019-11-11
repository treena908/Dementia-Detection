import pandas as pd
import numpy as np

def compute_confusion_average(df):
    matrix_sum = np.zeros((2,2))
    count = 0
    for row in df.itertuples():
        matrix_sum = matrix_sum + row.Confusion
        count = count + 1

    final_mat = matrix_sum/count
    print('TN: {}, FP:{}, FN:{}, TP:{}'.format(final_mat[0][0],final_mat[0][1],final_mat[1][0],final_mat[1][1]))


data = pd.read_pickle('../results/manual_feature_results111_attention_manual_class_weights6-0.pickle')
compute_confusion_average(data)
print(data.describe())



