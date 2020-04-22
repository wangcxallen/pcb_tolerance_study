# -*- coding: utf-8 -*-
"""
This script is used to train the autoencoder using the data generated from generate_tolerance_data.py
"""

# Libraries
import tensorflow
import keras
import numpy as np
# import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Dense, Conv2D, Dropout, BatchNormalization, Input, Reshape, Flatten, Deconvolution2D, Conv2DTranspose, MaxPooling2D, UpSampling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import adam
import pickle

# Utilities
def main():
    ae = tol_encoder()
    ae.train(5000)
    ae.save_weights('ae_weights')
    
    # Check training results
    # ae.load_weight('ae_weights')
    # raw_data = ae.train_data[60:61,:,:,0:]
    # prediction = ae.ae.predict(raw_data).reshape(28,28)
    
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # [X,Y] = np.meshgrid(np.arange(28),np.arange(28))
    # ax.plot_surface(X,Y,prediction,cmap='viridis', edgecolor='none')
    # plt.show()
    
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # [X,Y] = np.meshgrid(np.arange(28),np.arange(28))
    # ax.plot_surface(X,Y,raw_data.reshape(28,28),cmap='viridis', edgecolor='none')
    # plt.show()
    pass

class tol_encoder:
    '''
    '''
    def __init__(self):
        '''
        '''
        self.ae = self.create_ae()
        # compile it using adam optimizer
        self.ae.compile(optimizer="adam", loss="mse")
        # load data
        self.train_data = np.load('tol_data_1000.npy').reshape(-1,28,28,1)
        self.test_data = np.load('tol_data_50.npy').reshape(-1,28,28,1)
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
        l = Dense(49, activation='softmax')(l)
        
        #DECODER
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
    
    def train(self,num_epoch,batch_size=50):
        self.ae.fit(self.train_data, self.train_data, batch_size=batch_size, epochs=num_epoch)
    

# Main
if __name__ == '__main__':   
    main()