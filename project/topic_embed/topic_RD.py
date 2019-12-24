#%%
import pymssql #引入pymssql模块
import numpy as np
import pandas as pd

with pymssql.connect('127.0.0.1', None, None, 'dtest') as conn:
    with conn.cursor(as_dict=False) as cursor:
        sql = 'select topics, get_from from t_topic_embedden order by NEWID()'
        cursor.execute(sql)
        topics = []

        for i in range(4711147):
            row = cursor.fetchone()
            if row[1] == 1:
                indexs = [str(i[1:i.index(':')]) for i in row[0].split(',')]
            else:
                indexs = [str(i[1:]) for i in row[0].split(',')]
            topics.append(",".join(indexs))

        dict = {'topics': topics}
        df = pd.DataFrame(dict)
        df.to_csv("./topic_embed.csv", index=False, sep=' ')
