import keras
from keras.layers import Dense, Input
from keras.datasets import mnist
from keras.models import Model
import numpy as np
import numpy as np
import pandas as pd
from calls import SaveCall

class MGene(keras.utils.Sequence):
    dim = 100001

    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.data = pd.read_csv("./data/topic_embed.csv", sep=' ').values.reshape((-1,))
        self.len = len(self.data)

    def __getitem__(self, index):
        xs = np.zeros(shape=(self.batch_size, self.dim))
        base = index*batch_size
        for i in range(self.batch_size):
            indexs = [int(i) for i in self.data[i+base].split(',')]
            xs[i, indexs] = 1

        return xs, xs

    def __len__(self):
        return self.len // self.batch_size

    def on_epoch_end(self):
        np.random.shuffle(self.data)

input = Input(shape=(100001,))
#e1 = Dense(10000, activation='relu')(input)
embemdden = Dense(256, activation='relu')(input)
#d1 = Dense(10000, activation='relu')(embemdden)
output = Dense(100001, activation='sigmoid')(embemdden)

autoencoder = Model(input=input, output=output)
encoder = Model(input=input, output=embemdden)

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.summary()

batch_size = 256
gene = MGene(batch_size= batch_size)
save_call = SaveCall(filepath="./ckpt/{epoch}-{batch}-{loss:.6f}.ckpt", period=1000, mode=SaveCall.train_mode, max_one=False)
iepoch = save_call.load(autoencoder)
autoencoder.fit_generator(gene, epochs=20, verbose=2, shuffle=False, initial_epoch=iepoch, callbacks=[save_call], steps_per_epoch=gene.len//gene.batch_size)