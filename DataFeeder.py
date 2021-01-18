
import numpy as np
import keras
import random
import os
import pickle

class GeneratorFeeder(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self,baddatadir, dim, batch_size=8, shuffle=True):
        'Initialization'
        self.baddatadir  = baddatadir
        self.dim = dim
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.alldata = [os.path.join(baddatadir, d) for d in os.listdir(baddatadir)]
        self.labels  = np.ones(len(self.alldata))

        self.indexes = np.arange(len(self.alldata))
        random.shuffle(self.indexes)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.alldata) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        datainbatch = [self.alldata[k] for k in indexes]
        labelsinbatch = [self.labels[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(datainbatch,labelsinbatch)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            random.shuffle(self.indexes)

    def __data_generation(self, data,labels):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.dim[0],self.dim[1],self.dim[2]))
        Y = np.empty((self.batch_size), dtype=int)
        # Generate data
        for i, datafile in enumerate(data):
            # Store sample
            with open(datafile,"rb") as fh:
                X[i,] = np.expand_dims(pickle.load(fh),axis = 2)

            # Store class
            Y[i] = labels[i]

        return X, Y,

class PreTrainDiscrFeeder(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, gooddatadir,baddatadir, dim, batch_size=8, shuffle=True):
        'Initialization'
        self.gooddatadir = gooddatadir
        self.baddatadir  = baddatadir
        self.dim = dim
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.alldata = [os.path.join(gooddatadir, d) for d in os.listdir(gooddatadir)] + \
                       [os.path.join(baddatadir, d) for d in os.listdir(baddatadir)]
        self.labels  = np.zeros(len(self.alldata))
        for i in range(len(self.alldata)):
            if "good" in self.alldata[i]:
                self.labels[i] = 1

        self.indexes = np.arange(len(self.alldata))
        random.shuffle(self.indexes)
        print(self.indexes.shape)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.alldata) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        datainbatch = [self.alldata[k] for k in indexes]
        labelsinbatch = [self.labels[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(datainbatch,labelsinbatch)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            random.shuffle(self.indexes)

    def __data_generation(self, data,labels):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.dim[0],self.dim[1],self.dim[2]))
        Y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, datafile in enumerate(data):
            # Store sample
            with open(datafile,"rb") as fh:
                X[i,] = np.expand_dims(pickle.load(fh),axis = 2)

            # Store class
            Y[i] = labels[i]

        return X, Y,