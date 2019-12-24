import keras
from keras import Model, Input
from keras.layers import Dense, concatenate, PReLU, CuDNNGRU


ans_seq = 987
ans_dim = 15
member_dim = 19
member_topic_dim = 512
ques_title_dim = 256
ques_topic_dim = 256
ques_dim = ques_title_dim + ques_topic_dim + 1
#%%
# model
ans_ques_input = Input(shape=(ans_seq, ques_dim), name='ans_question_input')
ans_feat_input = Input(shape=(ans_seq, ans_dim), name='ans_feature_input')
member_feat_input = Input(shape = (member_dim,), name='member_feature_input')
member_topic_input = Input(shape=(member_topic_dim,), name='member_topic_input')
time_input = Input(shape = (1,), name = 'time_input')
ques_input = Input(shape = (ques_dim,), name='ques_input')

member_feat_dense = PReLU(name='member_feature_dense_prelu')(
    Dense(units=40, name='member_feature_dense')(member_feat_input))

member_topic_dense = PReLU(name='member_topic_dense_prelu')(
    Dense(units=256, name='member_topic_dense')(member_topic_input))

ans_feat_gru = CuDNNGRU(units=40, return_sequences=True, name='ans_feat_gru')(ans_feat_input)
ans_ques_gru = CuDNNGRU(units=256, return_sequences=True, name='ans_ques_gru')(ans_ques_input)

ans_con = concatenate([ans_feat_gru, ans_ques_gru], name='answer_concatenate')
answer_gru = CuDNNGRU(units=128, return_sequences=False, name='answer_gru')(ans_con)

question_dense = PReLU(name='ques_dense_prelu')(
    Dense(units=128, name='ques_dense')(ques_input))

time_dense = PReLU(name='time_dense_prelu')(
    Dense(units=5, name='time_dense')(time_input))

inv_con = concatenate([time_dense, member_feat_dense, member_topic_dense, answer_gru, question_dense], name='invite_concatenate')

inv_dense_1 = PReLU(name='inv_dense_1_prelu')(
    Dense(units=512, name='inv_dense_1')(inv_con))
inv_dense_2 = PReLU()(
    Dense(units=128, name='inv_dense_2_prelu')(inv_dense_1))
inv_out = Dense(units=1, activation='sigmoid', name='inv_out')(inv_dense_2)

model = Model(inputs = [time_input, ques_input, member_feat_input, member_topic_input, ans_feat_input, ans_ques_input], outputs = inv_out)
model.compile(optimizer='adam', loss=keras.losses.binary_crossentropy, metrics=['acc'])

model.summary()
keras.utils.plot_model(model, './model.png', show_shapes=True)