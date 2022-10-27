import numpy as np

class Canonical:
    """
    A class used to implement the DMP canonical system


    Attributes
    ----------
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
        step forward the canonical system (of dt sec) 
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
        """Reset phase attribute to 1."""
        self.s = 1


    def get_phase(self):
        """Get the value of the phase attribute.

        Returns
        -------
            float: phase value
        """
        return self.s


    def step(self, dt, dist = 0):
        """Step forward the canonical system

        Parameters
        ----------
        dt : float
            time step
        dist : float
            measured distance between commanded and real position

        """
        const = -self.alpha_phase / self.tau / (1 + self.alpha_stop * dist)
        self.s = self.s * np.exp(const * dt)
