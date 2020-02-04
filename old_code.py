# -*- coding: utf-8 -*-

# Libraries
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

# tolerance study
class Insertion:
    '''
    checkCollision
    tolerancePlotSurfaceSinglePoint
    tolerancePlotTriangleSinglePoint
    tolerancePlotSurfaceSinglePointMask
    tolerancePlotSurface
    sort
    tolerancePlotContour
    tolerancePlotPoint
    '''
    def __init__(self, x, y, Rh, Rp):
        '''
        x: array, (N,1)
        y: array, (N,1)
        Rh: array, (N,1) or scalar
        Rp: array, (N,1) or scalar 
        '''
        # Configuration
        self.x = x.reshape(-1) # (N,)
        self.y = y.reshape(-1) # (N,)
        self.Rh = Rh
        self.Rp = Rp
        pass
    
    def checkCollision(self,dx,dy,theta):
        '''
        Check collision of (dx,dy,theta) pairs on multi-pin component
        dx: array, (M,1)
        dy: array, (M,1)
        theta: array, (M,1)
        '''
        # Establish dx,dy,theta relationship
        xp = dx + np.multiply.outer(self.x, np.cos(theta)) - np.multiply.outer(self.y, np.sin(theta)) # (N,M)
        yp = dy + np.multiply.outer(self.x, np.sin(theta)) + np.multiply.outer(self.y, np.cos(theta)) # (N,M)
        # If all entries of axis0 of r <=0, then there is no collision. Otherwise there is collisions.
        r = (xp - self.x.reshape(-1,1))**2 + (yp - self.y.reshape(-1,1))**2 - (self.Rh - self.Rp)**2 # (N,M), braodcast is needed here
        
        return np.any(r>0.001, axis=0) # (M,)

    def tolerancePlotSurfaceSinglePoint(self):
        '''
        Successful
        '''
        # Configuration
        theta = np.linspace(-np.pi, np.pi, 100)
        phi = np.linspace(-np.pi, np.pi, 100)
        Theta, Phi = np.meshgrid(theta, phi)
        # Start figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # Compute dx,dy
        c1 = self.x*np.cos(Theta) - self.y*np.sin(Theta) - self.x # (100, 100)
        c2 = self.x*np.sin(Theta) + self.y*np.cos(Theta) - self.y # (100, 100)
        X = -c1 + (self.Rh-self.Rp)*np.cos(Phi) # (100, 100)
        Y = -c2 + (self.Rh-self.Rp)*np.sin(Phi) # (100, 100)
        Z = Theta
        # Plot
        ax.plot_surface(X, Y, Z)
        ax.set_xlabel('dx')
        ax.set_ylabel('dy')
        ax.set_zlabel('theta')

        plt.show()
        pass
    
    def tolerancePlotTriangleSinglePoint(self):
        '''
        Not working because of auto triangle matching
        '''
        # Configuration
        h = 50
        v = 50
        theta = np.linspace(-np.pi, np.pi, h)
        phi = np.linspace(-np.pi, np.pi, v)
        Theta, Phi = np.meshgrid(theta, phi)
        # Start figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('dx')
        ax.set_ylabel('dy')
        ax.set_zlabel('theta')
        # Compute points
        for i in np.arange(1):
            c1 = self.x[i]*np.cos(Theta) - self.y[i]*np.sin(Theta) - self.x[i] # (h, v)
            c2 = self.x[i]*np.sin(Theta) + self.y[i]*np.cos(Theta) - self.y[i] # (h, v)
            X = -c1 + (self.Rh-self.Rp)*np.cos(Phi) # (h, v)
            Y = -c2 + (self.Rh-self.Rp)*np.sin(Phi) # (h, v)
            Z = Theta.copy()
#            collision = self.checkCollision(X.reshape(-1), Y.reshape(-1), Z.reshape(-1)) # (h*v)
#            collision = collision.reshape(v,h)
#            idx = np.where(collision==True)
#            Z[idx] = np.NaN
            # Plot
            triang = mtri.Triangulation(X.flatten(), Y.flatten())
            ax.plot_trisurf(triang, Z.flatten(), cmap=plt.cm.CMRmap)

        plt.show()
        pass
    
    def tolerancePlotSurfaceSinglePointMask(self):
        '''
        Successful
        '''
        # Configuration
        theta = np.linspace(-np.pi, np.pi, 1000)
        phi = np.linspace(-np.pi, np.pi, 1000)
        Theta, Phi = np.meshgrid(theta, phi)
        # Start figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # Compute dx,dy
        c1 = self.x*np.cos(Theta) - self.y*np.sin(Theta) - self.x # (100, 100)
        c2 = self.x*np.sin(Theta) + self.y*np.cos(Theta) - self.y # (100, 100)
        X = -c1 + (self.Rh-self.Rp)*np.cos(Phi) # (100, 100)
        Y = -c2 + (self.Rh-self.Rp)*np.sin(Phi) # (100, 100)
        Z = Theta
        R = np.where(np.logical_and(Z>0.1,Z<0.6),Z,np.nan)
        # Plot
        ax.plot_surface(X, Y, R)
        ax.set_xlabel('dx')
        ax.set_ylabel('dy')
        ax.set_zlabel('theta')

        plt.show()
        
        pass
        
    def tolerancePlotSurface(self):
        '''
        Using Mask to plot the center area
        '''
        # Configuration
        h = 5000
        v = 500
        theta = np.linspace(-np.pi, np.pi, h)
        phi = np.linspace(-np.pi, np.pi, v)
        Theta, Phi = np.meshgrid(theta, phi)
        # Start figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('dx')
        ax.set_ylabel('dy')
        ax.set_zlabel('theta')
        # Compute points
        for i in np.arange(self.x.shape[0]):
            c1 = self.x[i]*np.cos(Theta) - self.y[i]*np.sin(Theta) - self.x[i] # (h, v)
            c2 = self.x[i]*np.sin(Theta) + self.y[i]*np.cos(Theta) - self.y[i] # (h, v)
            X = -c1 + (self.Rh-self.Rp)*np.cos(Phi) # (h, v)
            Y = -c2 + (self.Rh-self.Rp)*np.sin(Phi) # (h, v)
            Z = Theta.copy()
            collision = self.checkCollision(X.reshape(-1), Y.reshape(-1), Z.reshape(-1)) # (h*v)
            collision = collision.reshape(v,h)
            idx = np.where(collision==True)
            Z[idx] = np.NaN
            # Plot
            ax.plot_surface(X, Y, Z) #cmap="hot", vmin=-0.4, vmax=0.4

        plt.show()
    
    def tolerancePlotPoint(self):
        '''
        Currently for single point
        The interactive plot is really slow.
        '''
        # Configuration
        dx = np.linspace(-3, 3, 100)
        dy = np.linspace(-3, 3, 100)
        theta = np.linspace(-np.pi, np.pi, 100)
        # Start figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # Compute points
        flag = False
        num = 0
        for i in dx:
            for j in dy:
                for k in theta:
                    if num%10000==0:
                        print("%f Finished" % (num/1000000))
                    num +=1 
                    if np.all(self.checkCollision(i,j,k)==False):
                        if flag==False:
                            F = [np.array([i,j,k])]
                            flag = True
                        else:
                            F.append(np.array([i,j,k]))
        F = np.array(F)
        # Plot
        ax.scatter(F[:,0],F[:,1],F[:,2])
        ax.set_xlabel('dx')
        ax.set_ylabel('dy')
        ax.set_zlabel('theta')

        plt.show()
        pass
    
    def tryout(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        X = np.array([0,1,3,3,2,1])
        Y = np.array([0,1,1,1,1,0])
        Z = np.array([0,0,0,1,1,1])
        ax.plot_trisurf(X, Y, Z,shade=True)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()
        pass
    
    def sort(self, x, y):
        '''
        Sort the points in clockwise
        x: ndarray
        y: ndarray
        '''
        xm = np.mean(x)
        ym = np.mean(y)
        x = x - xm
        y = y - ym
        angle = np.arctan2(y,x)
        idx = np.argsort(angle)
        return idx
    
    def tolerancePlotContour(self, line=False, name="None"):
        '''
        The best currently, need to sort
        Using method in single point plot to obtain points on the surface
        Using checkCollision to check multiple constraints
        Sort all the points clockwise
        Plot the union volume through contour plot with width
        '''
        # Configuration
        Theta = np.linspace(-np.pi, np.pi, 1000) # (h,)
        phi = np.linspace(-np.pi, np.pi, 100) # (v,)
        # Obtain feasible surface points
        # Start figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        if line==True:
            for i in np.arange(self.x.shape[0]):
                x = self.x[i]*np.cos(Theta) - self.y[i]*np.sin(Theta) - self.x[i]
                y = self.x[i]*np.sin(Theta) + self.y[i]*np.cos(Theta) - self.y[i]
                ax.plot(x,y,Theta,c=(0.0,0.0,1.0),linewidth=1)
        for theta in Theta:
            pts = np.array([])
            # Obtain all surface points
            for i in np.arange(self.x.shape[0]):
                c1 = self.x[i]*np.cos(theta) - self.y[i]*np.sin(theta) - self.x[i] # (N, )
                c2 = self.x[i]*np.sin(theta) + self.y[i]*np.cos(theta) - self.y[i] # (N, )
                X = np.add.outer( -c1, (self.Rh-self.Rp)*np.cos(phi) ) # (N, v)
                X = X.reshape(-1) # (N*v,1)
                Y = np.add.outer( -c2, (self.Rh-self.Rp)*np.sin(phi) )# (N, v)
                Y = Y.reshape(-1) # (N*v,1)
                Z = theta*np.ones_like(X) # (N*v,1)
                # Check feasibility
                collision = self.checkCollision(X,Y,Z)
                idx = np.where(collision==False)
                if idx[0].size>0:
                    Xf = X[idx]
                    Yf = Y[idx]
                    Zf = Z[idx]
                    if i==0:
                        pts = np.stack( [Xf,Yf,Zf], axis=1)
                    else:
                        pts = np.concatenate((pts,np.stack( [Xf,Yf,Zf], axis=1)) ,axis=0)
                        
            if pts.size!=0:
                idx = self.sort(pts[:,1], pts[:,0])
                thetamin = -0.03
                thetamax = 0.03
                ax.plot(pts[idx,0], pts[idx,1], pts[idx,2], c=((theta-thetamin)/(thetamax-thetamin), (thetamax-theta)/(thetamax-thetamin), 0.2), linewidth=2)
        # Plot
        ax.set_xlabel('dx')
        ax.set_ylabel('dy')
        ax.set_zlabel('theta')
        ax.set_title(name)
        plt.show()
        pass
if __name__ == '__main__':   
    # Configuration Unit:mm
    com0 = {'name':'kf',
            'x':np.array([2.5*i for i in np.arange(8)]),
            'y':np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            'Rh':0.7,
            'Rp':0.4}# Rp must less than Rh
    
    com1 = {'name':'wj',
            'x':np.array([-5.08,0.0,5.08,-5.08,0.0,5.08]),
            'y':np.array([0.0,0.0,0.0,7.62,7.62,7.62]),
            'Rh':0.7,
            'Rp':0.5}# Rp must less than Rh
    
    com2 = {'name':'db',
            'x':np.array([1.0,-1.0,0.0]),
            'y':np.array([0.0,0.0,1.0]),
            'Rh':3.5,
            'Rp':3.0}# Rp must less than Rh
    
    com3 = {'name':'hx',
            'x':np.array([1.0,-1.0,0.0]),
            'y':np.array([0.0,0.0,1.0]),
            'Rh':3.5,
            'Rp':3.0}# Rp must less than Rh
    com = [com0,com1,com2,com3]
    
    for item in com[0:2]:
        singlePin = Insertion(item['x'], item['y'], item['Rh'], item['Rp'])
    #    singlePin.tolerancePlotSurfaceSinglePoint()
        singlePin.tolerancePlotContour(name=item['name'])