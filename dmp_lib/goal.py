import numpy as np

class Goal:
    """ Implementation of goal system for DMP """
    def __init__(self, g0, tau, alpha_g):
        if np.isscalar(g0):
            g0 = np.array([g0])

        self.g0 = g0
        self.tau = tau
        self.alpha_g = alpha_g
        # initialize goal
        self.g = g0


    def set_new_goal(self, new_g, force_goal=True):
        if np.isscalar(new_g):
            new_g = np.array([new_g])

        if new_g.shape == self.g0.shape:
            self.g0 = new_g
            if force_goal:
                self.g = new_g
        else:
            print('Goal dimensions not valid')


    def get_goal(self):
        return self.g


    def step(self, dt):
        dg = self.alpha_g*(self.g0-self.g)/self.tau
        self.g = self.g + dt*dg
