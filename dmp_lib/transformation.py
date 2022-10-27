import numpy as np

class Transformation:
    """ Implementation of transformation system for DMP """
    def __init__(self, x0, tau, K, D = None, style = 'advanced'):
        if np.isscalar(x0):
            x0 = np.array([x0])

        self.tau = tau
        self.K = K

        if D == None:
            self.D = 2*np.sqrt(K)
            
        self.x = x0
        self.x0 = x0
        self.z = np.zeros(x0.shape)
        self.style = style
        self.sys_dim = x0.shape[0]


    def reset(self, x0):
        if np.isscalar(x0):
            x0 = np.array([x0])

        if x0.shape == self.x.shape:
            self.x = x0
            self.x0 = x0
            self.z = np.zeros(x0.shape)
        else:
            print('Position dimensions not valid')


    def get_state(self):
        return self.x, self.z


    def step(self, dt, g, s, f=0):
        if self.style == 'advanced':
            dz = (self.K*(g-self.x)-self.D*self.z-self.K*(g-self.x0)*s+self.K*f)/self.tau
        elif self.style == 'classic':
            dz = (self.K*(g-self.x)-self.D*self.z+(g-self.x0)*f)/self.tau
        dx = self.z/self.tau
        
        self.x = self.x + dt*dx
        self.z = self.z + dt*dz
