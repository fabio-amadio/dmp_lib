import numpy as np

class Goal:
    """
    A class used to implement the DMP goal system.

    The goal system drives goal variable 'g' towards a desired 'g0'.

    Attributes
    ----------
    g : numpy.ndarray
        goal variable
    g0 : numpy.ndarray
        desired goal
    tau : float
        time scaling parameter
    alpha_g : float
        goal advancement parameter

    Methods
    -------
    set_new_goal(new_g, force_goal = True)
        reset phase to 1
    get_goal()
        get current goal value
    step(dt)
        step forward the canonical system (of dt sec) 
    """

    def __init__(self, g0, tau, alpha_g):
        """
        Parameters
        ----------
        g0 : numpy.ndarray
            desired goal
        tau : float
            time scaling parameter
        alpha_g : float
            goal advancement parameter
        """
        if np.isscalar(g0):
            g0 = np.array([g0])
        self.g0 = g0
        self.tau = tau
        self.alpha_g = alpha_g
        # initialize goal variable to desired goal
        self.g = g0


    def set_new_goal(self, new_g, force_goal=True):
        """Set new desired goal

        Parameters
        ----------
        new_g : numpy.ndarray
            new desired goal
        force_goal: bool, optional
            force goal variable to new desired goal (default is True)
        """
        if np.isscalar(new_g):
            new_g = np.array([new_g])

        if new_g.shape == self.g0.shape:
            self.g0 = new_g
            if force_goal:
                self.g = new_g
        else:
            print('Goal dimensions not valid')


    def get_goal(self):
        """Get current goal variable value.

        Returns
        -------
            float: goal value
        """
        return self.g


    def step(self, dt):
        """Step forward the goal system

        Parameters
        ----------
        dt : float
            time step
        """
        dg = self.alpha_g*(self.g0-self.g)/self.tau
        self.g = self.g + dt*dg
