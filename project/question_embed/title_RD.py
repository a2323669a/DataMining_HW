#%%
import pymssql #引入pymssql模块
import numpy as np
import pandas as pd

conn = pymssql.connect('127.0.0.1', None, None, 'dtest')
with conn.cursor(as_dict=False) as cursor:
    sql = 'select title_word_seq from t_question where title_word_seq <> \'-1\''
    cursor.execute(sql)
    titles = cursor.fetchall()
#%%
titles_vectors = []
count = 0
for title in titles:
    vectors = []
    words = title[0].split(',')
    for word in words:
        with conn.cursor()  as cursor:
            sql = 'select vectors from t_word where id = \'{}\''.format(word)
            cursor.execute(sql)
            vectors.append(np.array(cursor.fetchone()[0].split(' ')).astype(np.float16))
    titles_vectors.append(np.array(vectors))

    count += 1
    if count%2000 == 0:
        print("{}/{}".format(count, len(titles)))

del titles
np.save('./title_word.npy', titles_vectors)