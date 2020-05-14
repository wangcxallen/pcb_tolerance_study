# -*- coding: utf-8 -*-
"""
This script is used to train the autoencoder using the data generated from generate_tolerance_data.py
"""

# Libraries
from utilities_encoder import tol_encoder


# Utilities
def main():
    train_data = 'data/' + 'tol_data_10000_v1.npy'
    test_data = 'data/' + 'tol_data_50_v1.npy'
    ae = tol_encoder(train_data, test_data)
    ae.train(num_epoch=1, batch_size=50, filepath='weights-{epoch:02d}.hdf5')
    ae.save_weights('ae_weights')
    
    pass


# Main
if __name__ == '__main__':   
    main()