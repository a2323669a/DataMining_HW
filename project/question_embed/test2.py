#%%
import pymssql #引入pymssql模块
import numpy as np
import pandas as pd

conn = pymssql.connect('127.0.0.1', None, None, 'dtest')
with conn.cursor(as_dict=False) as cursor:
    sql = 'select title_word_seq from t_question where title_word_seq <> \'-1\''
    cursor.execute(sql)
    titles = cursor.fetchall()
counts = []
for title in titles:
    counts.append(len(title[0].split(',')))
#%%
counts = np.array(counts)
#%%
max = 38
import matplotlib.pyplot as plt

plt.hist(counts)
plt.savefig('./question_seq.png')
plt.show()
#%%
import numpy as np

b = np.load('./title_word.npy', allow_pickle=True)