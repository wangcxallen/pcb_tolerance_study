# -*- coding: utf-8 -*-
"""
This script is used to train the autoencoder using the data generated from generate_tolerance_data.py
"""

# Libraries
from utilities_encoder import tol_encoder


# Utilities
def main():
    train_data = 'data/' + 'tol_data_1000.npy'
    test_data = 'data/' + 'tol_data_50.npy'
    ae = tol_encoder(train_data, test_data)
    ae.train(num_epoch=5000)
    ae.save_weights('ae_weights')
    
    pass


# Main
if __name__ == '__main__':   
    main()