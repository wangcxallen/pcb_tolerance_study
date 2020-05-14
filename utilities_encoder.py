# -*- coding: utf-8 -*-
"""
Training utility: training class
"""

# Libraries
import tensorflow as tf
import keras
import numpy as np
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Dense, Conv2D, Dropout, BatchNormalization, Input, Reshape, Flatten, Deconvolution2D, Conv2DTranspose, MaxPooling2D, UpSampling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import adam
from keras.callbacks import ModelCheckpoint
import pickle

# Utilites
class tol_encoder:
    '''
    '''
    def __init__(self,train_data,test_data):
        '''
        '''
        self.ae = self.create_ae()
        # compile it using adam optimizer
        self.ae.compile(optimizer="adam", loss="mse")
        # load data
        self.train_data = np.load(train_data).reshape(-1,28,28,1)
        self.test_data = np.load(test_data).reshape(-1,28,28,1)
        pass 
    
    def create_ae(self):
        #ENCODER
        inp = Input((28, 28, 1))
        e = Conv2D(32, (3, 3), activation='relu')(inp)
        e = MaxPooling2D((2, 2))(e)
        e = Conv2D(64, (3, 3), activation='relu')(e)
        e = MaxPooling2D((2, 2))(e)
        e = Conv2D(64, (3, 3), activation='relu')(e)
        l = Flatten()(e)
        l = Dense(10, activation='softmax')(l)
        
        #DECODER
        l = Dense(49, activation='softmax')(l)
        d = Reshape((7,7,1))(l)
        d = Conv2DTranspose(64,(3, 3), strides=2, activation='relu', padding='same')(d)
        d = BatchNormalization()(d)
        d = Conv2DTranspose(64,(3, 3), strides=2, activation='relu', padding='same')(d)
        d = BatchNormalization()(d)
        d = Conv2DTranspose(32,(3, 3), activation='relu', padding='same')(d)
        decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(d)
        model = Model(inp, decoded)
        return model
    
    def load_weight(self,filename):
        filename = filename + '.pickle'
        with open(filename, 'rb') as handle:
            weights = pickle.load(handle)
        self.ae.set_weights(weights)
        pass
    
    def save_weights(self, filename):
        weights = self.ae.get_weights()
        filename = filename + '.pickle'
        with open(filename, 'wb') as handle:
            pickle.dump(weights, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pass
    
    def train(self, num_epoch,batch_size=20,filepath='weights-{epoch:02d}.hdf5'):   
        model_callback = ModelCheckpoint(filepath=filepath,
                                         save_weights_only=True,
                                         save_best_only=False)
        
        history = self.ae.fit(self.train_data, 
                              self.train_data, 
                              validation_data=(self.test_data,self.test_data), 
                              batch_size=batch_size, 
                              epochs=num_epoch,
                              callbacks=[model_callback])
        with open('history.pickle', 'wb') as handle:
            pickle.dump(history.history, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pass