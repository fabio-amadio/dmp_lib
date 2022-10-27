import numpy as np

class Canonical:
    """ Implementation of canonical system for DMP """
    def __init__(self, tau, alpha_phase, alpha_stop = 0):    
        self.tau = tau
        self.alpha_phase = alpha_phase
        self.alpha_stop = alpha_stop
        # initialize phase
        self.s = 1


    def reset(self):
        self.s = 1


    def get_phase(self):
        return self.s


    def step(self, dt, dist=0):
        const = -self.alpha_phase/self.tau/(1+self.alpha_stop*dist)
        self.s = self.s*np.exp(const*dt)
