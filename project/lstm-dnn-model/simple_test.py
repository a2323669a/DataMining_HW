import pandas as pd
import numpy as np
from typing import Dict
import keras
from keras import Model, Input
from keras.layers import CuDNNGRU, Dense, concatenate, PReLU, BatchNormalization
from calls import SaveCall

#load data
df_question = pd.read_csv('./data/ques.csv', sep=',')
df_invite = pd.read_csv('./data/invite.csv', sep=',')
df_invite_unlabeled = pd.read_csv('./data/invite_unlabeled.csv', sep=',')
df_member = pd.read_csv('./data/member.csv', sep=',')

t2_zero = np.load('./data/t2_zero.npy', allow_pickle=True).item()
title_zero :np.ndarray= t2_zero['title_zero']
topic_zero :np.ndarray= t2_zero['topic_zero']

# shuffle data and split test(30000) evaluation(10000)
df_invite.sample(frac=1, random_state=1, replace=True)
split_num = {'test' : 3e4, 'evaluation': 1.2e4}
evaluation_end = int(split_num['evaluation'])
test_end = int(split_num['evaluation'] + split_num['test'])

df_invite_evaluation = df_invite.iloc[0: evaluation_end]
df_invite_test = df_invite.iloc[evaluation_end: test_end]
df_invite = df_invite.iloc[test_end:]

# model parameter
ans_seq = 50
ans_dim = 15
member_dim = 19
member_topic_dim = 512
ques_title_dim = 256
ques_topic_dim = 256
ques_dim = ques_title_dim + ques_topic_dim + 1
#%%
def get_item(batch_size :int, index :int, invite :pd.DataFrame) -> Dict[str, np.ndarray]:
    time = np.zeros(shape=(batch_size,), dtype=np.float32)
    ques = np.zeros(shape=(batch_size, ques_dim), dtype=np.float32)
    member_feature = np.zeros((batch_size, member_dim), dtype=np.float32)
    is_answer = np.zeros(shape=(batch_size,), dtype=int)

    for i_batch in range(batch_size):
        t = invite.iloc[i_batch + index]

        #time
        time[i_batch] = float(t['time'])

        # question info
        q_id = t['q_id']
        series_ques = df_question[df_question['id'] == q_id].values[0]
        ques[i_batch] = np.hstack((
            float(series_ques[1]),
            np.array(series_ques[2].split(' ')).astype(np.float32),
            np.array(series_ques[3].split(' ')).astype(np.float32),
                          ))

        #member info
        inv_id = t['inv_id']
        series_member = df_member[df_member['id'] == inv_id].values[0]
        member_feature[i_batch] = np.hstack((
            np.array(series_member[1].split(' ')).astype(np.float32),
            series_member[2:]
        ))

        #is_answer
        is_answer[i_batch] = t['is_answer']


    data = {
        'time' :time,
        'question': ques,
        'member_feature': member_feature,
        'is_answer': is_answer
    }

    return data
#%%
# model
member_feat_input = Input(shape = (member_dim,), name='member_feature_input')
time_input = Input(shape = (1,), name = 'time_input')
ques_input = Input(shape = (ques_dim,), name='ques_input')

member_feature_bn = BatchNormalization(name='member_feature_bn')(member_feat_input)
time_bn = BatchNormalization(name='time_bn')(time_input)
ques_bn = BatchNormalization(name='question_bn')(ques_input)

member_feat_dense = PReLU(name='member_feature_dense_prelu')(
    Dense(units=40, name='member_feature_dense')(member_feature_bn))

question_dense = PReLU(name='ques_dense_prelu')(
    Dense(units=128, name='ques_dense')(ques_bn))

time_dense = PReLU(name='time_dense_prelu')(
    Dense(units=5, name='time_dense')(time_bn))

inv_con = concatenate([time_dense, member_feat_dense, question_dense], name='invite_concatenate')
inv_con_bn = BatchNormalization(name='inv_con_bn')(inv_con)

inv_dense_1 = PReLU(name='inv_dense_1_prelu')(
    Dense(units=512, name='inv_dense_1')(inv_con_bn))
inv_dense_2 = PReLU()(
    Dense(units=128, name='inv_dense_2_prelu')(inv_dense_1))
inv_out = Dense(units=1, activation='sigmoid', name='inv_out')(inv_dense_2)

model = Model(inputs = [time_input, ques_input, member_feat_input], outputs = inv_out)
model.compile(optimizer='adam', loss=keras.losses.binary_crossentropy, metrics=['acc'])

model.summary()
keras.utils.plot_model(model, './model.png', show_shapes=True)
#%%
class MGene(keras.utils.Sequence):
    def __init__(self, batch_size :int, invite :pd.DataFrame):
        self.batch_size = batch_size
        self.invite = invite
        self.length = len(invite)
        self.true_len_rate = len(invite[invite['is_answer'] == 1]) / self.length

    def on_epoch_end(self):
        self.invite.sample(frac=1, replace=True)

    def __getitem__(self, index):
        dict = get_item(self.batch_size, index, self.invite)

        return [
            dict['time'],
            dict['question'],
            dict['member_feature'],
        ], dict['is_answer']

    def __len__(self):
        return self.length // self.batch_size
#%%
train_gene = MGene(batch_size=64, invite=df_invite)

print('\nall data: {}\ntrain batch: {}'.format(len(df_invite), train_gene.__len__()))

save_call = SaveCall(filepath="./simple/ckpt/{epoch}-{batch}-{loss:.6f}.ckpt", period=30, mode=SaveCall.train_mode, max_one=False)
iepoch = save_call.load(model)

model.fit_generator(
    generator=train_gene, epochs=20, verbose=1, callbacks=[save_call],
    class_weight={0: train_gene.true_len_rate, 1: (1. - train_gene.true_len_rate)},
    shuffle=False, initial_epoch= iepoch
)