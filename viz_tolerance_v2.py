# -*- coding: utf-8 -*-
'''
This is the script to generate data that will be used by autoencoder.
Note:
    1. First thing that the autoencoder should know is to disgard frame shift. 
        Different frame may cause the tolerance to seem to be diifference but 
        they are actually the same socket.
'''
# Libraries
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

# Utilities
def main():
    # print('Start');
    get_data = get_tolerance_data('1_3_up',1)
    # test_data = get_data.get_data_batch()
    # print(test_data.shape)
    Z_up = np.load('1_3_up.npy')
    Z_down = np.load('1_3_down.npy')
    [X,Y] = np.meshgrid(get_data.delta_x,get_data.delta_y)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z_down[0,:,:],cmap='viridis_r', edgecolor='none')
    ax.plot_surface(X, Y, Z_up[0,:,:],cmap='viridis', edgecolor='none')
    ax.set_xlabel('delta_x')
    ax.set_ylabel('delta_y')
    ax.set_zlabel('theta')
    ax.set_title('1*3 socket')
    plt.show()
    pass
    
class get_tolerance_data:
    '''
    
    '''
    def __init__(self, filename, num_data, resolution=28):
        '''
        The data is now generated for nominal sockets from 2 to 20 single or double sets
        the radius tolerance is now randomly chosed from a certain range
        socket pin radius: 0.25
        socket pin distance: 2.5
        PCB board hole radius: 0.495
        
        Input:
            filename: where to store the data
            num_data: how much data you want
            resolution: resolution along x or y axis of the tolerance space
        '''
        # Configuration
        self.filename = filename
        self.num_data = num_data
        self.res = resolution
        self.dp=2.5
        self.Rp=0.25
        self.Rh=0.495
        self.tol_r = self.Rh - self.Rp
        self.delta_x = np.linspace(-self.tol_r, self.tol_r, self.res) # these are the check points along delta_x
        self.delta_y = np.linspace(-self.tol_r, self.tol_r, self.res)
        pass
    
    def get_data_batch(self):
        '''
        generate 2d tolerance data and store them into filename.npz
        '''
        data = []
        for i_data in np.arange(self.num_data):
            print('Generating data: %d' % (100*i_data/self.num_data))
            tolerance = self.get_2d_tolerance()
            data.append(tolerance)
        
        data = np.array(data)
        np.save(self.filename,data)
        return data
    
    def get_2d_tolerance(self, res_theta=1000):
        '''
        randomly choose distribution of pins within the socket
        the distribution should have a reference of the frame in the center
        Input:
            res_theta: resolution along theta axis
        Ouput:
            tolerance_2d: array, (m, m), where m is the resolution along x or y dimension in tolerance space
        '''
        # Configure pin distribution and radius tolerance randomly and generate theta axis test points
        num_row = 1 #np.random.randint(1,3)
        num_col = 3 #np.random.randint(2,21)
        xp_p_frame = 1.25 * np.array( num_row * [i for i in np.arange(num_col)] )
        yp_p_frame = 2.5 * np.multiply.outer(np.arange(num_row),np.ones(num_col)).reshape(-1,)
        x_center = np.average(xp_p_frame)
        y_center = np.average(yp_p_frame)
        xp_p_frame = xp_p_frame - x_center
        yp_p_frame = yp_p_frame - y_center
        theta = np.linspace(0,3.1415,res_theta) # for up
        # theta = np.linspace(-3.1415,0,res_theta) # for down
        
        self.tol_r = 0.24 # + 0.05*np.random.random()
        
        # Calculate tolerance 2d
        is_collision = self.check_collision(xp_p_frame,yp_p_frame,theta) # (m, m, n), data type: bool
        # is_collision = np.flip(is_collision,axis=2) # for down
        tolerance_2d = np.argmax(is_collision,axis=2) # (m, m)
        tolerance_2d =  (tolerance_2d) * (3.1415/res_theta) # for up
        # tolerance_2d = - (tolerance_2d) * (3.1415/res_theta) # for down
        return tolerance_2d
    
    def check_collision(self,x,y,theta,th=0.0000001):
        '''
        Check collision of multi-pin component x, y at (delta_x,delta_y,theta) pairs that specified by input theta and member para delta_x,delta_y
        Input:
            x: array, (N,1), coodinates of pins in p_frame, which should be centered first
            y: array, (N,1)
            theta: array, (n,1), discrete the theta dimension
        Output:
            is_collision: array, (m, m, n), where m is the resolution along x or y dimension in tolerance space
        '''
        # Configure vec_delta_x,vec_delta_y ,and vec_theta to be the correct shape
        [delta_X,delta_Y] = np.meshgrid(self.delta_x,self.delta_y) # (m,m), 28*28 for MNIST like data
        vec_delta_x = delta_X.reshape(-1) # (m*m,)
        vec_delta_y = delta_Y.reshape(-1) # (m*m,)
        
        vec_theta = np.multiply.outer(np.ones_like(vec_delta_x), theta) # (m*m,n)
        vec_theta = vec_theta.reshape(-1) # (m*m*n,) i.e. (M,)
        
        vec_delta_x = np.multiply.outer(vec_delta_x, np.ones_like(theta)) # (m*m,n)
        vec_delta_x = vec_delta_x.reshape(-1) # (m*m*n,) i.e. (M,)
        vec_delta_y = np.multiply.outer(vec_delta_y, np.ones_like(theta)) # (m*m,n)
        vec_delta_y = vec_delta_y.reshape(-1) # (m*m*n,) i.e. (M,)
        
        # Compute is_collision vector
        xp_h_frame = vec_delta_x + np.multiply.outer(x, np.cos(vec_theta)) - np.multiply.outer(y, np.sin(vec_theta)) # (N,M)
        yp_h_frame = vec_delta_y + np.multiply.outer(x, np.sin(vec_theta)) + np.multiply.outer(y, np.cos(vec_theta)) # (N,M)
        # If all entries of axis0 of r <=0, then there is no collision. Otherwise there is collisions.
        d = (xp_h_frame - x.reshape(-1,1))**2 + (yp_h_frame - y.reshape(-1,1))**2 - (self.tol_r)**2 # (N,M), braodcast is needed here
        is_collision = np.any(d>th, axis=0)
        is_collision = is_collision.reshape(self.res, self.res, np.size(theta)) # (m, m, n)
        return  is_collision # (m, m, n)
    
# Main
if __name__ == '__main__':   
    main()