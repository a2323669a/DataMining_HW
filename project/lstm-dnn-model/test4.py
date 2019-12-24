import pandas as pd
import numpy as np

df = pd.read_csv('./data/unlabeled.csv', encoding='utf-16', sep=';')
df_dict = dict(zip(df.columns.values.astype(str), list(range(len(df.columns.values)))))
data = df.values
del df

t2_zero = np.load('./data/t2_zero.npy', allow_pickle=True).item()
title_zero :np.ndarray= t2_zero['title_zero']
topic_zero :np.ndarray= t2_zero['topic_zero']
ques_zero = np.hstack((np.zeros((1,1)),title_zero, topic_zero))

# model parameter
ans_seq = 50
ans_dim = 15
member_dim = 19
member_topic_dim = 512
ques_title_dim = 256
ques_topic_dim = 256
ques_dim = ques_title_dim + ques_topic_dim + 1
#%%
from typing import Dict
def getitem(index :int, batch_size :int) ->Dict[str, np.ndarray]:
    t_data = data[index :batch_size + index]

    time = t_data[:, df_dict['time']]

    question = np.array([np.array(i[df_dict['ques_feature']].split(' ')).astype(np.float32) for i in t_data])

    member_feature = np.array([np.array(i[df_dict['member_feature']][:-1].split(' ')).astype(np.float32) for i in t_data])

    member_topic = np.array([np.array(i[df_dict['topic']].split(' ')).astype(np.float32) for i in t_data])

    answer = np.array([np.array(i[df_dict['ans_ques']].split(',')) for i in t_data])
    ans_feature = np.zeros((batch_size,ans_seq,ans_dim))
    ans_ques = ques_zero.repeat(ans_seq, axis=0).reshape((1,ans_seq,ques_dim)).repeat(batch_size, axis = 0)
    for idx in range(ans_feature.shape[0]):
        if answer[idx][0] != '-1':
            ans_feature[idx, :len(answer[idx]), :] = np.array(
                [np.array(i[:i.index('|')] .split(' ')).astype(np.float32) for i in answer[idx]])[:50]
            ans_ques[idx, :len(answer[idx]), :] = np.array(
                [np.array(i[i.index('|')+1:].split(' ')).astype(np.float32) for i in answer[idx]])[:50]

    data_dict = {
        'time': time,
        'question': question,
        'member_feature': member_feature,
        'member_topic': member_topic,
        'ans_feature': ans_feature,
        'ans_question': ans_ques,
    }

    return data_dict
#%%
k = getitem(0, 128)