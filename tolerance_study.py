# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-

# Libraries
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# tolerance study
class Insertion:
    '''
    checkCollision
    sort
    tolerancePlotContour
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
    
    def checkCollision(self,x,y,theta,dx=None,dy=None, th=0.0000001):
        '''
        Check collision of (x,y,theta) pairs on multi-pin component with defection (dx,dy)
        x: array, (M,1)
        y: array, (M,1)
        theta: array, (M,1)
        dx: vector, (N,)
        dy: vector, (N,)
        '''
        # Configuration
        if np.any(dx)==None:
            dx = np.zeros_like(self.x)
        if np.any(dy)==None:
            dy = np.zeros_like(self.y)
        # Establish dx,dy,theta relationship
        xp = x + np.multiply.outer(self.x+dx, np.cos(theta)) - np.multiply.outer(self.y+dy, np.sin(theta)) # (N,M)
        yp = y + np.multiply.outer(self.x+dx, np.sin(theta)) + np.multiply.outer(self.y+dy, np.cos(theta)) # (N,M)
        # If all entries of axis0 of r <=0, then there is no collision. Otherwise there is collisions.
        d = (xp - self.x.reshape(-1,1))**2 + (yp - self.y.reshape(-1,1))**2 - (self.Rh - self.Rp)**2 # (N,M), braodcast is needed here
        return np.any(d>th, axis=0) # (M,)
    
    def checkFeasible(self,x,y,theta,dx=None,dy=None, th=-0.0000001):
        '''
        Check collision of (x,y,theta) pairs on multi-pin component with defection (dx,dy)
        x: array, (M,1)
        y: array, (M,1)
        theta: array, (M,1)
        dx: vector, (N,)
        dy: vector, (N,)
        '''
        # Configuration
        if np.any(dx)==None:
            dx = np.zeros_like(self.x)
        if np.any(dy)==None:
            dy = np.zeros_like(self.y)
        # Establish dx,dy,theta relationship
        xp = x + np.multiply.outer(self.x+dx, np.cos(theta)) - np.multiply.outer(self.y+dy, np.sin(theta)) # (N,M)
        yp = y + np.multiply.outer(self.x+dx, np.sin(theta)) + np.multiply.outer(self.y+dy, np.cos(theta)) # (N,M)
        # If all entries of axis0 of r <=0, then there is no collision. Otherwise there is collisions.
        d = (xp - self.x.reshape(-1,1))**2 + (yp - self.y.reshape(-1,1))**2 - (self.Rh - self.Rp)**2 # (N,M), braodcast is needed here
        return np.all(d<th, axis=0) # (M,)
    
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
    
    def single_plot(self, dx=None, dy=None, line=False, name="Name"):
        '''
        The best currently, need to sort
        Using method in single point plot to obtain points on the surface
        Using checkCollision to check multiple constraints
        Sort all the points clockwise
        Plot the union volume through contour plot with width
        '''
        # Configuration
        if np.any(dx)==None:
            dx = np.zeros_like(self.x)
        if np.any(dy)==None:
            dy = np.zeros_like(self.y)
        Theta = np.linspace(-np.pi, np.pi, 1000) # (h,)
        phi = np.linspace(-np.pi, np.pi, 100) # (v,)
        xp = dx + self.x
        yp = dy + self.y
        xh = self.x
        yh = self.y
        # Obtain feasible surface points
        # Start figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        if line==True:
            for i in np.arange(self.x.shape[0]):
                x = xp[i]*np.cos(Theta) - yp[i]*np.sin(Theta) - xh[i]
                y = xp[i]*np.sin(Theta) + yp[i]*np.cos(Theta) - yh[i]
                ax.plot(x,y,Theta,c=(0.0,0.0,1.0),linewidth=1)
        for theta in Theta:
            pts = np.array([])
            # Obtain all surface points
            for i in np.arange(self.x.shape[0]):
                c1 = xp[i]*np.cos(theta) - yp[i]*np.sin(theta) - xh[i] # (N, )
                c2 = xp[i]*np.sin(theta) + yp[i]*np.cos(theta) - yh[i] # (N, )
                X = np.add.outer( -c1, (self.Rh-self.Rp)*np.cos(phi) ) # (N, v)
                X = X.reshape(-1) # (N*v,1)
                Y = np.add.outer( -c2, (self.Rh-self.Rp)*np.sin(phi) )# (N, v)
                Y = Y.reshape(-1) # (N*v,1)
                Z = theta*np.ones_like(X) # (N*v,1)
                # Check feasibility
                collision = self.checkCollision(X,Y,Z,dx,dy)
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
    
    def double_plot(self, dx, dy, name='Name'):
        '''
        dx, dy: ndarray, defection of every pin
        '''
        # Configuration
        Theta = np.linspace(-np.pi, np.pi, 1000) # (h,)
        phi = np.linspace(-np.pi, np.pi, 100) # (v,)
        xp = dx + self.x
        yp = dy + self.y
        # Obtain feasible surface points
        # Start figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for theta in Theta:
            pts = np.array([])
            pts_new = np.array([])
            # Obtain all surface points
            for i in np.arange(self.x.shape[0]):
                c1 = self.x[i]*np.cos(theta) - self.y[i]*np.sin(theta) - self.x[i] # (N, )
                c2 = self.x[i]*np.sin(theta) + self.y[i]*np.cos(theta) - self.y[i] # (N, )
                X = np.add.outer( -c1, (self.Rh-self.Rp)*np.cos(phi) ) # (N, v)
                X = X.reshape(-1) # (N*v,)
                Y = np.add.outer( -c2, (self.Rh-self.Rp)*np.sin(phi) )# (N, v)
                Y = Y.reshape(-1) # (N*v,)
                Z = theta*np.ones_like(X) # (N*v,)
                # Check feasibility
                collision = self.checkCollision(X,Y,Z)
                idx = np.where(collision==False)
                if idx[0].size>0:
                    Xf = X[idx] # (Ns,)
                    Yf = Y[idx] # (Ns,)
                    Zf = Z[idx] # (Ns,)
                    if pts.size==0:
                        pts = np.stack( [Xf,Yf,Zf], axis=1)
                    else:
                        pts = np.concatenate((pts,np.stack( [Xf,Yf,Zf], axis=1)) ,axis=0)
                    
        
            for i in np.arange(self.x.shape[0]):
                c1 = xp[i]*np.cos(theta) - yp[i]*np.sin(theta) - self.x[i] # (N, )
                c2 = xp[i]*np.sin(theta) + yp[i]*np.cos(theta) - self.y[i] # (N, )
                X = np.add.outer( -c1, (self.Rh-self.Rp)*np.cos(phi) ) # (N, v)
                X = X.reshape(-1) # (N*v,1)
                Y = np.add.outer( -c2, (self.Rh-self.Rp)*np.sin(phi) )# (N, v)
                Y = Y.reshape(-1) # (N*v,1)
                Z = theta*np.ones_like(X) # (N*v,1)
                # Check feasibility
                collision = self.checkCollision(X,Y,Z,dx,dy)
                idx = np.where(collision==False)
                if idx[0].size>0:
                    Xf = X[idx] # (Ns,)
                    Yf = Y[idx] # (Ns,)
                    Zf = Z[idx] # (Ns,)
                    if pts_new.size==0:
                        pts_new = np.stack( [Xf,Yf,Zf], axis=1)
                    else:
                        pts_new = np.concatenate((pts_new,np.stack( [Xf,Yf,Zf], axis=1)) ,axis=0)
            # Plot
            # Contour for the new volume, Red
            if pts_new.size!=0:
                idx = self.sort(pts_new[:,1], pts_new[:,0])
                ax.plot(pts_new[idx,0], pts_new[idx,1], pts_new[idx,2], c=(1.0,0,0), linewidth=2)
            # Contour for the non-defected volume, colorful
            if pts.size!=0:
                idx = self.sort(pts[:,1], pts[:,0])
                thetamin = -0.03
                thetamax = 0.03
                ax.plot(pts[idx,0], pts[idx,1], pts[idx,2], c=(0.2, (theta-thetamin)/(thetamax-thetamin), (thetamax-theta)/(thetamax-thetamin)), linewidth=2)
        
        # Plot configuration
        ax.set_xlabel('dx')
        ax.set_ylabel('dy')
        ax.set_zlabel('theta')
        ax.set_title(name)
        plt.show()
        pass
    
    def difference_plot(self, dx, dy, name='Name'):
        '''
        dx, dy: ndarray, defection of every pin
        '''
        # Configuration
        Theta = np.linspace(-np.pi, np.pi, 1000) # (h,)
        phi = np.linspace(-np.pi, np.pi, 100) # (v,)
        xp = dx + self.x
        yp = dy + self.y
        # Obtain feasible surface points
        # Start figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for theta in Theta:
            # Analyze surface of old feasible domain
            pts_short = np.array([])
            pts_new = np.array([])
            thetamin = -0.2
            thetamax = 0.2
            
            for i in np.arange(self.x.shape[0]):
                c1 = self.x[i]*np.cos(theta) - self.y[i]*np.sin(theta) - self.x[i] # (N, )
                c2 = self.x[i]*np.sin(theta) + self.y[i]*np.cos(theta) - self.y[i] # (N, )
                X = np.add.outer( -c1, (self.Rh-self.Rp)*np.cos(phi) ) # (N, v)
                X = X.reshape(-1) # (N*v,)
                Y = np.add.outer( -c2, (self.Rh-self.Rp)*np.sin(phi) )# (N, v)
                Y = Y.reshape(-1) # (N*v,)
                Z = theta*np.ones_like(X) # (N*v,)
                # Check feasibility
                collision = self.checkCollision(X,Y,Z)
                idx = np.where(collision==False)
                if idx[0].size>0:
                    Xf = X[idx] # (Ns,)
                    Yf = Y[idx] # (Ns,)
                    Zf = Z[idx] # (Ns,)
                    # Check defection condition
                    collision = self.checkCollision(Xf,Yf,Zf,dx,dy) # (Ns,)
                    feasible = self.checkFeasible(Xf,Yf,Zf,dx,dy) # (Ns,)
                    con1 = np.where(collision==True) # (N1,)
                    con2 = np.where(feasible==True) # (N2,)
                    if con2[0].size>0:
                        Xn = Xf[con2] # (Nn,)
                        Yn = Yf[con2] # (Nn,)
                        Zn = Zf[con2] # (Nn,)
                        ax.plot(Xn, Yn, Zn, c=(1.0,0,0), linewidth=2)
                        if pts_new.size==0:
                            pts_new = np.stack( [Xn,Yn,Zn], axis=1)
                        else:
                            pts_new = np.concatenate((pts_new,np.stack( [Xn,Yn,Zn], axis=1)) ,axis=0) # (Ns_new1, 3)
                    
                    if con1[0].size>0:
                        Xs = Xf[con1] # (Ns,)
                        Ys = Yf[con1] # (Ns,)
                        Zs = Zf[con1] # (Ns,)
                        ax.plot(Xs, Ys, Zs, c=(0.0, (theta-thetamin)/(thetamax-thetamin), (thetamax-theta)/(thetamax-thetamin)), linewidth=2)

                        if pts_short.size==0:
                            pts_short = np.stack( [Xs,Ys,Zs], axis=1)
                        else:
                            pts_short = np.concatenate((pts_short,np.stack( [Xs,Ys,Zs], axis=1)) ,axis=0) # (Ns_short1, 3)
            # Plot
            # Contour for the short volume, colorful
            # if pts_short.size!=0:
            #     idx = self.sort(pts_short[:,1], pts_short[:,0])
            #     thetamin = -0.03
            #     thetamax = 0.03
            #     ax.plot(pts_short[idx,0], pts_short[idx,1], pts_short[idx,2], c=(0.0, (theta-thetamin)/(thetamax-thetamin), (thetamax-theta)/(thetamax-thetamin)), linewidth=2)
            # Contour for the new volume, Red
            # if pts_new.size!=0:
            #     idx = self.sort(pts_new[:,1], pts_new[:,0])
            #     ax.plot(pts_new[idx,0], pts_new[idx,1], pts_new[idx,2], c=(1.0,0,0), linewidth=2)
                
            
            # Analyze surface of new feasible domain
            pts_short = np.array([])
            pts_new = np.array([])
            for i in np.arange(self.x.shape[0]):
                c1 = xp[i]*np.cos(theta) - yp[i]*np.sin(theta) - self.x[i] # (N, )
                c2 = xp[i]*np.sin(theta) + yp[i]*np.cos(theta) - self.y[i] # (N, )
                X = np.add.outer( -c1, (self.Rh-self.Rp)*np.cos(phi) ) # (N, v)
                X = X.reshape(-1) # (N*v,1)
                Y = np.add.outer( -c2, (self.Rh-self.Rp)*np.sin(phi) )# (N, v)
                Y = Y.reshape(-1) # (N*v,1)
                Z = theta*np.ones_like(X) # (N*v,1)
                # Check feasibility
                collision = self.checkCollision(X,Y,Z,dx,dy)
                idx = np.where(collision==False)
                if idx[0].size>0:
                    Xf = X[idx] # (Ns,)
                    Yf = Y[idx] # (Ns,)
                    Zf = Z[idx] # (Ns,)
                    # Check defection condition
                    collision = self.checkCollision(Xf,Yf,Zf) # (Ns,)
                    feasible = self.checkFeasible(Xf,Yf,Zf) # (Ns,)
                    con1 = np.where(collision==True) # (N1,)
                    con2 = np.where(feasible==True) # (N2,)
                    if con1[0].size>0:
                        Xn = Xf[con1] # (Nn,)
                        Yn = Yf[con1] # (Nn,)
                        Zn = Zf[con1] # (Nn,)
                        ax.plot(Xn, Yn, Zn, c=(1.0,0.0,0.0), linewidth=2)
                        if pts_new.size==0:
                            pts_new = np.stack( [Xn,Yn,Zn], axis=1)
                        else:
                            pts_new = np.concatenate((pts_new,np.stack( [Xn,Yn,Zn], axis=1)) ,axis=0) # (Ns_new1, 3)
                    
                    if con2[0].size>0:
                        Xs = Xf[con2] # (Ns,)
                        Ys = Yf[con2] # (Ns,)
                        Zs = Zf[con2] # (Ns,)
                        ax.plot(Xs, Ys, Zs, c=(0.0, (theta-thetamin)/(thetamax-thetamin), (thetamax-theta)/(thetamax-thetamin)), linewidth=2)
                        if pts_short.size==0:
                            pts_short = np.stack( [Xs,Ys,Zs], axis=1)
                        else:
                            pts_short = np.concatenate((pts_short,np.stack( [Xs,Ys,Zs], axis=1)) ,axis=0) # (Ns_short1, 3)
            
            # Plot
            # Contour for the short volume, colorful
            # if pts_short.size!=0:
            #     idx = self.sort(pts_short[:,1], pts_short[:,0])
            #     thetamin = -0.03
            #     thetamax = 0.03
            #     ax.plot(pts_short[idx,0], pts_short[idx,1], pts_short[idx,2], c=(0.0, (theta-thetamin)/(thetamax-thetamin), (thetamax-theta)/(thetamax-thetamin)), linewidth=2)
            # Contour for the new volume, Red
            # if pts_new.size!=0:
            #     idx = self.sort(pts_new[:,1], pts_new[:,0])
            #     ax.plot(pts_new[idx,0], pts_new[idx,1], pts_new[idx,2], c=(1.0,0,0), linewidth=2)
        
        # Plot configuration
        ax.set_xlabel('dx')
        ax.set_ylabel('dy')
        ax.set_zlabel('theta')
        ax.set_title(name)
        plt.show()
        pass
    
    def center_plot2D(self, dx, dy):
        # Configuration
        Theta = np.linspace(-np.pi, np.pi, 1000) # (h,)
        xp = dx + self.x
        yp = dy + self.y
        xh = self.x
        yh = self.y
        # Plot
        fig, ax = plt.subplots()
        for i in np.arange(self.x.shape[0]):
            c1 = - (xp[i]*np.cos(Theta) - yp[i]*np.sin(Theta) - xh[i]) # -dx@0
            c2 = - (xp[i]*np.sin(Theta) + yp[i]*np.cos(Theta) - yh[i]) # -dy@0
            ax.plot(c1,c2,linewidth=1,label=i)
            ax.legend()
        pass
    
if __name__ == '__main__':   
    # Configuration Unit:mm
    com0 = {'name':'kf',
            'x':np.array([2.5*i for i in np.arange(8)]),
            'y':np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            'dx':np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            'dy':np.array([0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0]),
            'Rh':0.7,
            'Rp':0.4}# Rp must less than Rh
    
    com1 = {'name':'wj',
            'x':np.array([-2.5,0.0,2.5,-2.5,0.0,2.5]),
            'y':np.array([0.0,0.0,0.0,2.5,2.5,2.5]),
            'dx':np.array([0.0,0.0,0.0,0.0,0.0,0.0]),
            'dy':np.array([0.0,0.0,0.0,0.0,0.0,0.0]),
            'Rh':0.7,
            'Rp':0.4}# Rp must less than Rh
    
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
    
    for item in com[1:2]:
        insertion = Insertion(item['x'], item['y'], item['Rh'], item['Rp'])
        # insertion.double_plot(item['dx'], item['dy'], name=item['name'])
        insertion.difference_plot(item['dx'], item['dy'], name=item['name'])
        # insertion.center_plot2D(item['dx'], item['dy'])
