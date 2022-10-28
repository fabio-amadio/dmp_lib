"""canonical module

This module implements the Canonical class.
"""

import numpy as np

class Canonical:
    """Canonical system class.

    A class used to implement the DMP canonical system.
    The canonical system drives phase variable 's' from 1 to 0, 
    dictating the time for the other systems composing a DMP.

    Attributes
    ----------
    s : float
        phase variable
    tau : float
        time scaling parameter
    alpha_phase : float
        phase decay parameter
    alpha_stop : float
        phase stop parameter

    Methods
    -------
    reset()
        reset phase to 1
    get_phase()
        get phase value
    step(dt, dist = 0)
        single-step canonical system update (of dt sec) 
    """

    def __init__(self, tau, alpha_phase, alpha_stop = 0):
        """
        Parameters
        ----------
        tau : float
            time scaling parameter
        alpha_phase : float
            phase decay parameter
        alpha_stop : float, optional
            phase stop parameter (default is 0)
        """
        self.tau = tau
        self.alpha_phase = alpha_phase
        self.alpha_stop = alpha_stop
        self.s = 1


    def reset(self):
        """Reset phase attribute to 1.
        """
        self.s = 1


    def get_phase(self):
        """Get current phase variable value.

        Returns
        -------
        float
            phase value
        """
        return self.s


    def step(self, dt, dist = 0):
        """Single-step canonical system update.

        Parameters
        ----------
        dt : float
            time step
        dist : float, optional
            distance between commanded and real position (default is 0)
        """
        const = -self.alpha_phase / self.tau / (1 + self.alpha_stop * dist)
        self.s = self.s * np.exp(const * dt)
        