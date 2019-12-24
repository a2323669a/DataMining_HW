import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from keras.layers.recurrent import GRU, LSTM
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential
from keras.layers.core import Dense, RepeatVector
from keras.models import Model
import numpy as np
import keras
from calls import SaveCall

class MGene(keras.utils.Sequence):
    maxlen = 38
    stop = 37.0
    dim = 64
    path = './data/title_word.npy'

    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.words = np.load(self.path, allow_pickle=True)
        np.random.shuffle(self.words)

    def __getitem__(self, index):
        xs = np.ones(shape=(self.batch_size,self.maxlen, self.dim), dtype='float16') * self.stop

        for i in range(self.batch_size):
            x = self.words[i + index * self.batch_size]
            xs[i, 0:x.shape[0], :] = x

        return xs, xs


    def __len__(self):
        return self.words.shape[0] // self.batch_size

    def on_epoch_end(self):
        np.random.shuffle(self.words)

embed_size = 256

model = Sequential()
model.add(LSTM(input_dim=MGene.dim, output_dim=128, return_sequences=True))
model.add(LSTM(units=embed_size, return_sequences=False, name='encode'))
model.add(RepeatVector(MGene.maxlen))
model.add(LSTM(embed_size, return_sequences=True))
model.add(LSTM(units=128, return_sequences=True))
model.add(TimeDistributed(Dense(output_dim=MGene.dim, activation="linear")))

model.compile(loss="mse", optimizer='adam')

encoder = Model(inputs = model.input, outputs = model.get_layer(name='encode').output)

gene = MGene(batch_size=32)
save_call = SaveCall(filepath="./ckpt/{epoch}-{batch}-{loss:.6f}.ckpt", period=1500, mode=SaveCall.train_mode, max_one=False)
iepoch = save_call.load(model)
model.fit_generator(gene, epochs=20, initial_epoch=iepoch, shuffle=False, callbacks=[save_call], verbose=0)