# Copyright 2022 by Fabio Amadio.
# All rights reserved.
# This file is part of the dmp_lib,
# and is released under the "GNU General Public License".
# Please see the LICENSE file included in the package.

import numpy as np

class Transformation:
    """
    A class used to implement the DMP transformation system.
    The transformation system drives the DMP state 'x' towards a goal
    'g' following a mass-spring-damper dynamics perturbed by a 
    nonlinear forcing term.
    The 'classic' formulation refers to the one in Schaal et al., 2006.
    The 'advanced' formulation refers to the one in Pastor et al., 2009.

    Attributes
    ----------
    x : numpy.ndarray
        state variable
    z : numpy.ndarray
        scaled velocity variable
    x0 : numpy.ndarray
        initial state
    sys_dim: int
        system dimension
    tau : float
        time scaling parameter
    K : float
        stiffness parameter
    D : float
        damping parameter
    style: string
        dynamics style ('advanced' or 'classic')
    """

    def __init__(self, x0, tau, K, D = None, style = 'advanced'):
        """
        Parameters
        ----------
        x0 : numpy.ndarray
            initial state
        tau : float
            time scaling parameter
        K : float
            stiffness parameter
        D : float, optional
            damping parameter (default is None -> D = 2 * sqrt(K))
        style: string, optional
            dynamics style (default is 'advanced')
        """
        if np.isscalar(x0):
            x0 = np.array([x0])

        self.tau = tau
        self.K = K

        if D == None:
            self.D = 2 * np.sqrt(K)
            
        self.x = x0
        self.x0 = x0
        self.z = np.zeros(x0.shape)
        self.style = style
        self.sys_dim = x0.shape[0]


    def reset(self, x0):
        """Reset transformation system to initial state x0.
        
        Parameters
        ----------
        x0 : numpy.ndarray
            initial state
        """
        if np.isscalar(x0):
            x0 = np.array([x0])

        if x0.shape == self.x.shape:
            self.x = x0
            self.x0 = x0
            self.z = np.zeros(x0.shape)
        else:
            print('Position dimensions not valid')


    def get_state(self):
        """Get current state.

        Returns
        -------
        list
            a list of numpy.ndarray containing x and z
        """
        return self.x, self.z


    def step(self, dt, g, s, f = 0):
        """Single-step transformation system update.

        Parameters
        ----------
        dt : float
            time step
        g : numpy.ndarray
            goal position
        s : float
            phase variable
        f : numpy.ndarray, optional
            forcing term (default is 0)
        """
        if self.style == 'advanced':
            dz = (self.K * (g - self.x) - self.D * self.z - self.K * \
                (g - self.x0) * s + self.K * f) / self.tau
        elif self.style == 'classic':
            dz = (self.K * (g - self.x) - self.D * self.z + \
                (g - self.x0) * f) / self.tau

        dx = self.z / self.tau
        
        self.x = self.x + dt * dx
        self.z = self.z + dt * dz
