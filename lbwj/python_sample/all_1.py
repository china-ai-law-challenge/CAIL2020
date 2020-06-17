import pandas as pd
import jieba
import numpy as np
import sys
import os

if __name__ == '__main__':
    input_path = '/input'
    output_path = '/output'
    input_filename = 'SMP-CAIL2020-test1.csv'
    output_filename = 'result1.csv'
    input_file = input_path + '/' + input_filename
    output_file = output_path + '/' + output_filename

    df1 = pd.read_csv(input_file)
    df2 = pd.DataFrame(columns=['id','answer'])

    for i in range(len(df1)):
        df2.loc[i,'id'] = df1.loc[i,'id']
        df2.loc[i,'answer'] = 1

    df2['id'] = df2['id'].astype('int')
    df2['answer'] = df2['answer'].astype('int')
    df2.to_csv(output_file,encoding='utf-8',index=False)
