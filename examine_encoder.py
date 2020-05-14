# -*- coding: utf-8 -*-
"""
This script is used to examine the training results
"""
# import libraries
import pickle
import numpy as np
import matplotlib.pyplot as plt
from utilities_encoder import tol_encoder

# Utilities
def main():
    train_data = 'data/' + 'tol_data_1000.npy'
    test_data = 'data/' + 'tol_data_50.npy'
    ae = tol_encoder(train_data, test_data)
    
    # viz_history('history_10_5000')
    viz_prediction(ae,'ae_10_5000')
    pass

def viz_history(filename):
    '''
    filename: filename of the history
    '''
    prefix = filename
    filename = 'results/data/' + filename + '.pickle'
    with open(filename, 'rb') as handle:
            hist = pickle.load(handle)
    val_loss = np.array(hist['val_loss'])
    loss = np.array(hist['loss'])
    epoch = np.arange(loss.shape[0])
    fig = plt.figure()
    plt.plot(epoch,np.log(val_loss),label=prefix+' test loss')
    plt.plot(epoch,np.log(loss),label=prefix+' loss')
    plt.legend()
    plt.title(prefix +' log')
    fig.savefig('results/image/'+prefix)
    
    
def viz_prediction(ae_model, filename):
    '''
    filename: filename of the ae weights
    '''
    # Check training results
    prefix = filename
    filename = 'results/data/' + filename
    ae = ae_model
    ae.load_weight(filename)
    idx = np.random.randint(ae.train_data.shape[0])
    raw_data = ae.train_data[idx:(idx+1),:,:,0:]
    prediction = ae.ae.predict(raw_data).reshape(28,28)
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    [X,Y] = np.meshgrid(np.arange(28),np.arange(28))
    ax.plot_surface(X,Y,prediction,cmap='viridis', edgecolor='none')
    ax.set_title(filename+' prediction')
    plt.show()
    fig.savefig('results/image/'+prefix+'_pred')
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    [X,Y] = np.meshgrid(np.arange(28),np.arange(28))
    ax.plot_surface(X,Y,raw_data.reshape(28,28),cmap='viridis', edgecolor='none')
    ax.set_title(filename+' ground truth')
    plt.show()
    fig.savefig('results/image/'+prefix+'_gt')
    pass

# Main
if __name__ == '__main__':   
    main()

