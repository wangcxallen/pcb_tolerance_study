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
from mpl_toolkits import mplot3d # mlot3d and plt are for data testing
import matplotlib.pyplot as plt

# global read only variables
DISTANCE_COL = 2.5 # No randomization introduced, distance in x direction
DISTANCE_ROW = 2.5 # No randomization introduced, distance in y direction
ROW_MAX = 2
ROW_MIN = 1
COL_MAX = 20
COL_MIN = 2

RADIUS_HOLE = 0.495
RADIUS_PIN = 0.25
RADIUS_TOL_RANGE = 0.1 # Half range of randomized radius tolerance, radius tolerance = Rh-Rp


# Utilities
def main():
    print('Start');
    get_data = get_tolerance_data('test_data',1)
    get_data.get_data_batch()
    
    test_data = np.load('test_data.npy')[0]
    [X,Y] = np.meshgrid(get_data.delta_x,get_data.delta_y)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, test_data,cmap='viridis', edgecolor='none')
    ax.set_title('Surface plot')
    plt.show()
    pass
    
class get_tolerance_data:
    '''
    
    '''
    def __init__(self, filename, num_data, resolution=28):
        '''
        Input:
            filename: where to store the data
            num_data: how much data you want
            resolution: resolution along x or y axis of the tolerance space
        '''
        # Configuration
        self.filename = filename
        self.num_data = num_data
        self.res = resolution
        self.d_row = DISTANCE_ROW
        self.d_col = DISTANCE_COL
        self.Rp = RADIUS_PIN
        self.Rh = RADIUS_HOLE
        self.tol_r = RADIUS_HOLE - RADIUS_PIN + RADIUS_TOL_RANGE # in order to include the most loose constrait
        self.delta_x = np.linspace(-self.tol_r, self.tol_r, self.res) # these are the check points along delta_x
        self.delta_y = np.linspace(-self.tol_r, self.tol_r, self.res)
        pass
    
    def get_data_batch(self):
        '''
        generate 2d tolerance data and store them into filename.npz
        '''
        data = []
        data_config = []
        for i_data in np.arange(self.num_data):
            print('Generating data: %f %%' % (100*i_data/self.num_data))
            [tolerance, config] = self.get_2d_tolerance()
            data.append(tolerance)
            data_config.append(config)
        
        data = np.array(data)
        data_config = np.array(data_config)
        np.save(self.filename,data)
        np.save(self.filename+'_config',data_config)
        pass
    
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
        num_row = np.random.randint(1,3)
        num_col = np.random.randint(2,21)
        xp_p_frame = self.d_col * np.array( num_row * [i for i in np.arange(num_col)] )
        yp_p_frame = self.d_row * np.multiply.outer(np.arange(num_row),np.ones(num_col)).reshape(-1,)
        x_center = np.average(xp_p_frame)
        y_center = np.average(yp_p_frame)
        xp_p_frame = xp_p_frame - x_center
        yp_p_frame = yp_p_frame - y_center
        theta = np.linspace(0,np.pi,res_theta)
        
        self.tol_r = RADIUS_HOLE - RADIUS_PIN + RADIUS_TOL_RANGE*(np.random.random()*2-1)
        
        # Calculate tolerance 2d
        is_collision = self.check_collision(xp_p_frame,yp_p_frame,theta) # (m, m, n), data type: bool
        tolerance_2d = np.argmax(is_collision,axis=2) # (m, m)
        tolerance_2d = (tolerance_2d) * (np.pi/res_theta)
        
        # Record configuration
        config = [num_row, num_col, self.tol_r, ]
        return [tolerance_2d,config]
    
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