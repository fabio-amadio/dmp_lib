import numpy as np
import numpy.matlib as matlib
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
from scipy.signal import savgol_filter
from dmp_lib.transformation import Transformation
from dmp_lib.canonical import Canonical
from dmp_lib.goal import Goal
from dmp_lib.geometry import quat_to_rot
from dmp_lib.geometry import rot_to_quat
from dmp_lib.geometry import rot_to_axis_angle
from dmp_lib.geometry import axis_angle_to_rot
from dmp_lib.geometry import axis_angle_to_quat
from dmp_lib.geometry import quat_to_axis_angle
from dmp_lib.geometry import solve_discontinuities
from dmp_lib.geometry import is_rotation_matrix


class DMP():
    """
    A class used to implement the DMP comprising of a canonical, goal 
    and transformation system working together.

    Attributes
    ----------
    canonical_sys : dmp_lib.canonical.Canonical
        canonical system

    goal_sys : dmp_lib.goal.Goal
        goal system

    transformation_sys : dmp_lib.transformation.Transformation
        transformation system

    z : numpy.ndarray
        scaled velocity variable

    x0 : numpy.ndarray
        initial state

    sys_dim: int
        system dimension

    tau : float
        time scaling parameter

    dt : float
        sampling interval

    num_basis : int
        number of basis functions

    centers: numpy.ndarray
        basis function centers

    widths: numpy.ndarray
        basis function widths

    weights: numpy.ndarray
        basis function weights
    """

    def __init__(self, num_basis, x0, g0, dt, tau,
                 alpha_phase = 25 / 3, alpha_stop = 25, alpha_g = 25 / 2, 
                 K = 25**2 / 2, D = None, style = 'advanced'):
        """
        Parameters
        ----------
        num_basis : int
            number of basis functions

        x0 : numpy.ndarray
            initial state

        g0 : numpy.ndarray
            goal state

        dt : float
            sampling time

        tau : float
            time scaling parameter

        alpha_phase : float, optional
            phase decay parameter (default is 25 / 3)

        alpha_stop : float, optional
            phase stop parameter (default is 0)

        alpha_g : float, optional
            goal advancement parameter (default is 25 / 2)

        K : float, optional
            stiffness parameter (default is 25**2 / 2)

        D : float, optional
            damping parameter (default is None -> D = 2 * sqrt(K))

        style: string, optional
            dynamics style (default is 'advanced')
        """
        if np.isscalar(x0):
            x0 = np.array([x0])
        if np.isscalar(g0):
            g0 = np.array([g0])

        # save system dimension
        self.sys_dim = x0.size

        # init canonical system
        self.canonical_sys = Canonical(tau, alpha_phase, alpha_stop)

        # init goal system
        self.goal_sys = Goal(g0, tau, alpha_g)

        # init transformation system
        self.transformation_sys = Transformation(x0, tau, K, D, style)

        self.tau = tau
        self.dt = dt
        self.num_basis = num_basis

        # init forcing
        self.init_forcing(num_basis)


    def init_forcing(self, num_basis):
        """Initialize parameters of the forcing term.

        Parameters
        ----------
        num_basis : int
            number of basis functions
        """
        self.centers = np.zeros((num_basis, self.sys_dim))
        self.widths = np.zeros((num_basis, self .sys_dim))
        self.weights = np.zeros((num_basis, self.sys_dim))

        # compute Gaussian centers
        c_locations = np.exp(-self.canonical_sys.alpha_phase * \
            np.array(list(range(num_basis))) / (num_basis - 1))
        self.centers = \
            matlib.repmat(np.expand_dims(c_locations, 1), 1, self.sys_dim)

        # compute Gaussian widths
        for k in range(num_basis-1):
            self.widths[k,:] = 1 / (self.centers[k+1,:] - self.centers[k,:])**2 
        self.widths[-1,:] = self.widths[-2,:]


    def evaluate_basis(self, s):
        """Evaluate basis functions at phase s.

        Parameters
        ----------
        s : float
            phase location

        Returns
        -------
        numpy.ndarray
            basis functions values in s
        """
        if np.size(s) == 1:
            return np.exp(-self.widths*(s - self.centers)**2)


    def evaluate_forcing(self, s):
        """Evaluate forcing term at phase s.

        Parameters
        ----------
        s : float
            phase location

        Returns
        -------
        numpy.ndarray
            forcing term in s
        """
        all_phi_s = self.evaluate_basis(s)
        den = np.sum(all_phi_s, 0)
        if np.sum(den) > 0.:
            f = np.sum(all_phi_s*self.weights,0)/den*s
        else:
            f = np.zeros(self.sys_dim)
        return f


    def evaluate_weighted_basis(self, s):
        """Evaluate weighted basis functions at phase s.

        Parameters
        ----------
        s : float
            phase location
            
        Returns
        -------
        numpy.ndarray
            weighted sum of basis functions values in s
        """
        all_phi_s = self.evaluate_basis(s)
        return all_phi_s * self.weights


    def reset(self, x0):
        """Reset DMP at initial state x0.

        Parameters
        ----------
        x0 : numpy.ndarray
            initial state
        """
        if np.isscalar(x0):
            x0 = np.array([x0])

        self.canonical_sys.reset() # reset phase
        self.transformation_sys.reset(x0) # reset transformation system


    def set_tau(self, new_tau):
        """Set time scaling parameter.

        Parameters
        ----------
        new_tau: float
            new time scaling parameter
        """
        self.tau = new_tau
        self.canonical_sys.tau = new_tau
        self.transformation_sys.tau = new_tau
        self.goal_sys.tau = new_tau


    def set_new_goal(self, new_g, force_goal = True):
        """Set new desired goal.

        Parameters
        ----------
        new_g : numpy.ndarray
            new goal 

        force_goal : bool, optional
            choose if override g variable or only g0 (default is True)
        """
        self.goal_sys.set_new_goal(new_g, force_goal=force_goal) # update goal system


    def get_phase(self):
        """Get current phase from canonical system.

        Returns
        -------
        float
            phase value
        """
        return self.canonical_sys.get_phase()


    def get_goal(self):
        """Get current goal from goal system.
        
        Returns
        -------
        numpy.ndarray
            goal value
        """
        return self.goal_sys.get_goal()


    def get_output(self):
        """Get DMP output from transformation system.

        Returns
        -------
        list
            a list with current DMP position and velocity
        """
        [x, z] = self.transformation_sys.get_state()
        return x, z / self.tau


    def step(self, dist = 0):
        """Single-step DMP update.

        Parameters
        ----------
        dist : float

        """
        s = self.get_phase()
        g = self.get_goal()
        f = self.evaluate_forcing(s)

        self.transformation_sys.step(self.dt, g, s, f)
        self.canonical_sys.step(self.dt, dist=dist)
        self.goal_sys.step(self.dt)


    def learn_from_demo(self, x_demo, time_steps, filter_mode = 'savgol'):
        """Fit forcing function to reproduce desired trajectory.

           Attributes tau, x0, g0 are modified according to demo.

        Parameters
        ----------
        x_demo : numpy.ndarray
            demo trajectory [x_demo.shape : (num_steps, sys_dim)]

        time_steps : numpy.ndarray
            time steps [time_steps.shape : (num_steps,)]

        filter_mode : string, optional
            either 'savgol' or 'cd' (default is 'savgol')

        Returns
        -------
        list
            list with (processed) x_demo, dx_demo, ddx_demo, time_steps

        Raises
        ------
        ValueError
            if x_demo and time_steps have different length
        """
        x_demo = np.reshape(x_demo, (-1, self.sys_dim))
        time_steps = time_steps-time_steps[0] # initial time to 0 sec

        if x_demo.shape[0] != time_steps.shape[0]:
            raise ValueError("x_demo and time_steps must have the same length")

        # interpolate demo
        path_gen = interpolate.interp1d(time_steps, x_demo.T)
        # use constant sampling intervals
        time_steps = np.arange(0., time_steps[-1], self.dt)
                
        num_steps = len(time_steps)

        # modiffy tau
        self.set_tau(num_steps*self.dt)

        x_demo = path_gen(time_steps).T

        if filter_mode == 'savgol':
            # compute derivatives with Savitzky-Golay filter
            x_demo = savgol_filter(x_demo, 15, 3, axis = 0, mode = 'nearest')
            dx_demo = savgol_filter(x_demo, 15, 3, axis = 0, mode = 'nearest',
                                    deriv = 1, delta = self.dt)
            ddx_demo = savgol_filter(x_demo, 15, 3, axis = 0, mode = 'nearest',
                                     deriv = 2, delta = self.dt)

        elif filter_mode == 'cd':
            dx_demo = np.zeros(x_demo.shape)
            ddx_demo = np.zeros(x_demo.shape)
            # compute derivatives with central difference
            for i in range(self.sys_dim):
                dx_demo[:,i] = np.gradient(x_demo[:,i], time_steps)
                ddx_demo[:,i] = np.gradient(dx_demo[:,i], time_steps)

        g = x_demo[-1,:]

        # compute phase values
        s_demo = np.zeros(num_steps)
        s_demo[0] = self.canonical_sys.get_phase()
        for k in range(1, num_steps):
            dt = time_steps[k] - time_steps[k-1]
            self.canonical_sys.step(dt)
            s_demo[k] = self.canonical_sys.get_phase()
        self.canonical_sys.reset()       

        # get forcing term weights for each system dimension
        f_d_all = np.zeros((self.sys_dim, num_steps))
        for j in range(self.sys_dim):
            # compute desired forcing term
            if self.transformation_sys.style == 'advanced':
                f_d = self.tau**2 * ddx_demo[:,j] / self.transformation_sys.K \
                    -(g[j] - x_demo[:,j]) + self.tau \
                    * self.transformation_sys.D / self.transformation_sys.K \
                    *dx_demo[:,j] + (g[j] - x_demo[0,j]) * s_demo

            elif self.transformation_sys.style == 'classic':
                if g[j]==x_demo[0,j]:
                    f_d = np.zeros(x_demo[:,j].shape)
                else:
                    f_d = (self.tau**2 * ddx_demo[:,j] - \
                        self.transformation_sys.K * (g[j] - x_demo[:,j]) + \
                        self.tau * self.transformation_sys.D * \
                        dx_demo[:,j]) / (g[j] - x_demo[0,j])  
            f_d_all[j,:] = f_d

        for j in range(self.sys_dim):
            f_d = f_d_all[j:j+1,:].T
            # get regressor matrix
            Phi = np.zeros((num_steps, self.num_basis))
            for k in range(num_steps):
                basis = self.evaluate_basis(s_demo[k])[:,j] 
                Phi[k,:] = basis / np.sum(basis)*s_demo[k]

            # Solve: Phi * w = f_d
            fitted_w = np.dot(np.linalg.pinv(Phi), f_d)
            self.weights[:,j] = fitted_w[:,0]


        return x_demo, dx_demo, ddx_demo, time_steps


    def get_weights(self):
        """Get DMP weights.

        Returns
        -------
        numpy.ndarray
            basis functions weights
        """
        return self.weights


    def import_weights(self, new_weights):
        """Import DMP weights.

        Parameters
        ----------
        new_weights : numpy.ndarray
            new DMP weights [new_weights.shape = (num_basis, sys_dim)]
        """
        if new_weights.shape == self.weights.shape:
            self.weights = new_weights
        else:
            print("Weights dimensions not valid")

    def rollout(self):
        """Run a DMP rollout and reset.

        Returns
        -------
        list
            list of numpy.array describing the rollout
        """
        time_steps = np.arange(0., self.tau, self.dt)

        g0 = self.get_goal()
        s = np.zeros((len(time_steps), 1))
        x = np.zeros((len(time_steps), self.sys_dim))
        v = np.zeros((len(time_steps), self.sys_dim))
        g = np.zeros((len(time_steps), self.sys_dim))

        forcing = np.zeros((len(time_steps), self.sys_dim))

        for k, t in enumerate(time_steps):
            s[k] = self.get_phase()
            x[k,:], v[k,:] = self.get_output()
            g[k,:] = self.get_goal()
            forcing[k,:] = self.evaluate_forcing(s[k])
            self.step()

        self.reset(self.transformation_sys.x0)

        return time_steps, x, v, g, forcing, s

    def info(self):
        """Return info about DMP type.

        Returns
        -------
        string
            DMP type
        """
        return 'single'


class Bimanual_DMP(DMP):
    """
    A class used to implement the Bimanual DMP that controls a virtual 
    frame placed between the two end-effectors. Each end-effector pose 
    is defined by a fixed rotation and variable translation from the 
    central virtual frame. Bimanual_DMP extends the main DMP class.

    Attributes
    ----------
    canonical_sys : dmp_lib.canonical.Canonical
        canonical system

    goal_sys : dmp_lib.goal.Goal
        goal system

    transformation_sys : dmp_lib.transformation.Transformation
        transformation system

    z : numpy.ndarray
        scaled velocity variable

    x0 : numpy.ndarray
        initial state

    sys_dim: int
        system dimension

    tau : float
        time scaling parameter

    dt : float
        sampling interval

    num_basis : int
        number of basis functions

    centers: numpy.ndarray
        basis function centers

    widths: numpy.ndarray
        basis function widths

    weights: numpy.ndarray
        basis function weights

    R_right : numpy.ndarray
        right EE rotation matrix [R_right.shape = (3,3)]

    R_left : numpy.ndarray
        left EE rotation matrix [R_left.shape = (3,3)]
    """

    def __init__(self, num_basis, x0, g0, dt, tau,
                 alpha_phase = 25 / 3, alpha_stop = 25, alpha_g = 25 / 2,
                 K = 25**2 / 2, D = None, style = 'advanced',
                 R_right = np.eye(3), R_left = np.eye(3)):
        """
        Parameters
        ----------
        num_basis : int
            number of basis functions

        x0 : numpy.ndarray
            initial state

        g0 : numpy.ndarray
            goal state

        dt : float
            sampling time

        tau : float
            time scaling parameter

        alpha_phase : float, optional
            phase decay parameter (default is 25 / 3)

        alpha_stop : float, optional
            phase stop parameter (default is 0)

        alpha_g : float, optional
            goal advancement parameter (default is 25 / 2)

        K : float, optional
            stiffness parameter (default is 25**2 / 2)

        D : float, optional
            damping parameter (default is None -> D = 2 * sqrt(K))

        style: string, optional
            dynamics style (default is 'advanced')

        R_right: numpy.ndarray, optional
            right EE rotation matrix (default is np.eye(3))

        R_left: numpy.ndarray, optional
            left EE rotation matrix (default is np.eye(3))

        Raises
        ------
        ValueError
            if R_right or R_left are not proper rotation matrices
            if len(x0) is less than 7 (virtual frame prose + distance)

        """
        if x0.size < 7:
            raise ValueError('len(x0) must be at least 7 \
                (virtual frame prose + distance)')

        if not is_rotation_matrix(R_right):
            raise ValueError('R_right is not a rotation matrix')

        if not is_rotation_matrix(R_left):
            raise ValueError('R_left is not a rotation matrix')

        DMP.__init__(self,num_basis, x0, g0, dt, tau,
            alpha_phase, alpha_stop, alpha_g, K, D, style)

        self.R_right = R_right
        self.R_left = R_left


    def get_poses(self):
        """Get end-effector positions from virtual frame pose.

        Returns
        -------
        list
            list with left and right positions and quaternions 
        """
        # DMP state: virtual frame pose & distance (+ other components not used here)
        x,_ = self.get_output()

        vf_position = x[0:3]
        vf_axis_angle = x[3:6]
        vf_R = axis_angle_to_rot(vf_axis_angle)
        dist = x[6]

        # world-to-virtual-frame
        H_vf = np.zeros((4,4))
        H_vf[0:3,0:3] = vf_R
        H_vf[0:3,3] = vf_position
        H_vf[3,3] = 1.0

        # virtual-frame-to-right
        vf_H_r = np.zeros((4,4))
        vf_H_r[0:3,0:3] = self.R_right
        vf_H_r[0:3,3] = np.array([0,0,-dist/2])
        vf_H_r[3,3] = 1.0

        # virtual-frame-to-left
        vf_H_l = np.zeros((4,4))
        vf_H_l[0:3,0:3] = self.R_left
        vf_H_l[0:3,3] = np.array([0,0,dist/2])
        vf_H_l[3,3] = 1.0

        # compose transformations
        H_r = np.matmul(H_vf, vf_H_r)
        H_l = np.matmul(H_vf, vf_H_l)

        R_r = H_r[0:3,0:3]
        q_r = np.array(rot_to_quat(R_r))
        p_r = H_r[0:3,3]

        R_l = H_l[0:3,0:3]
        q_l = np.array(rot_to_quat(R_l))
        p_l = H_l[0:3,3]

        return p_r, q_r, p_l, q_l


    def rollout(self):
        """Run a DMP rollout and reset.

        Returns
        -------
        list
            list of numpy.array describing the bimanual rollout
        """
        time_steps = np.arange(0., self.tau, self.dt)
        
        g0 = self.get_goal()
        s = np.zeros((len(time_steps), 1))
        x = np.zeros((len(time_steps), self.sys_dim))
        v = np.zeros((len(time_steps), self.sys_dim))
        g = np.zeros((len(time_steps), self.sys_dim))

        forcing = np.zeros((len(time_steps), self.sys_dim))

        p_r = np.zeros((len(time_steps), 3))
        q_r = np.zeros((len(time_steps), 4))
        p_l = np.zeros((len(time_steps), 3))
        q_l = np.zeros((len(time_steps), 4))
        for k, t in enumerate(time_steps):
            s[k] = self.get_phase()
            x[k,:], v[k,:] = self.get_output()
            g[k,:] = self.get_goal()
            forcing[k,:] = self.evaluate_forcing(s[k])
            p_r[k,:], q_r[k,:], p_l[k,:], q_l[k,:] = self.get_poses()
            self.step()

        self.reset(self.transformation_sys.x0)

        return time_steps, p_r, q_r, p_l, q_l, x, v, g, forcing, s

    def info(self):
        """Return info about DMP type.

        Returns
        -------
        string
            DMP type
        """
        return 'dual'



class Target_DMP(DMP):
    """Target-Referred DMP subclass (extension of DMP).
    A class used to implement the Target-Referred DMP (TR-DMP) that
    learned demo trajectory w.r.t. a target reference frame that can
    be changed at execution time.

    Attributes
    ----------
    canonical_sys : dmp_lib.canonical.Canonical
        canonical system
    goal_sys : dmp_lib.goal.Goal
        goal system
    transformation_sys : dmp_lib.transformation.Transformation
        transformation system
    z : numpy.ndarray
        scaled velocity variable
    x0 : numpy.ndarray
        initial state
    sys_dim: int
        system dimension
    tau : float
        time scaling parameter
    dt : float
        sampling interval
    num_basis : int
        number of basis functions
    centers: numpy.ndarray
        basis function centers
    widths: numpy.ndarray
        basis function widths
    weights: numpy.ndarray
        basis function weights
    w_H_t: numpy.ndarray
        world-to-target homogeneous transformation
    """

    def __init__(self, num_basis, x0, g0, dt, tau,
                 alpha_phase = 25 / 3, alpha_stop = 25, alpha_g = 25 / 2, 
                 K = 25**2 / 2, D = None, style = 'advanced',
                 w_H_t = np.eye(4)):
        """
        Parameters
        ----------
        num_basis : int
            number of basis functions

        x0 : numpy.ndarray
            initial state

        g0 : numpy.ndarray
            goal state

        tau : float
            time scaling parameter

        dt : float
            sampling time

        alpha_phase : float, optional
            phase decay parameter (default is 25 / 3)

        alpha_stop : float, optional
            phase stop parameter (default is 0)

        alpha_g : float, optional
            goal advancement parameter (default is 25 / 2)

        K : float, optional
            stiffness parameter (default is 25**2 / 2)

        D : float, optional
            damping parameter (default is None -> D = 2 * sqrt(K))

        style: string, optional
            dynamics style (default is 'advanced')

        w_H_t: numpy.ndarray, optional
            world-to-target hom. transformation (default is np.eye(4))

        Raises
        ------
        ValueError
            if w_H_t is not a proper homogeneous transformation matrix
        """
        if x0.size < 6:
            raise ValueError('x0 must represent a 6D end-effector pose')

        DMP.__init__(self, num_basis, x0, g0, dt, tau,
            alpha_phase, alpha_stop, alpha_g, K, D, style)

        if w_H_t.shape != (4,4):
            if not is_rotation_matrix(w_H_t[:3,:3]):
                if not (w_H_t[3,0] == w_H_t[3,1] == w_H_t[3,2] == 0. 
                    and w_H_t[3,3] == 1): 
                    raise ValueError('w_H_t is not a proper \
                        homogeneous transformation matrix')

        self.w_H_t = w_H_t

    def learn_from_demo(self, x_demo, time_steps, filter_mode = 'cd'):
        """Fit forcing function to reproduce desired trajectory.

           Attributes tau, x0, g0 are modified according to demo.

        Parameters
        ----------
        x_demo : numpy.ndarray
            demo trajectory [x_demo.shape : (num_steps, sys_dim)]

        time_steps : numpy.ndarray
            time steps [time_steps.shape : (num_steps,)]

        filter_mode : string, optional
            either 'savgol' or 'cd' (default is 'savgol')

        Returns
        -------
        list
            list with (processed) x_demo, dx_demo, ddx_demo, time_steps

        Raises
        ------
        ValueError
            if x_demo and time_steps have different length
        """
        x_demo = np.reshape(x_demo, (-1, self.sys_dim))
        time_steps = time_steps-time_steps[0] # initial time to 0 sec

        # interpolate demo
        path_gen = interpolate.interp1d(time_steps, x_demo.T)

        # use constant sampling intervals
        time_steps = np.arange(0., time_steps[-1], self.dt)
        
        num_steps = len(time_steps)

        # set tau equal to demo time length
        self.set_tau(num_steps*self.dt)

        x_demo = path_gen(time_steps).T

        # get demo in target reference frame
        axis_list = []
        angle_list = []
        for k in range(x_demo.shape[0]):       

            w_position_ee = x_demo[k,0:3]
            w_axis_angle_ee = x_demo[k,3:6]
            w_R_ee = axis_angle_to_rot(w_axis_angle_ee)  

            # world-to-end-effector
            w_H_ee = np.zeros((4,4))
            w_H_ee[0:3,0:3] = w_R_ee
            w_H_ee[0:3,3] = w_position_ee
            w_H_ee[3,3] = 1.0

            # target-to-end-effector
            t_H_ee = np.matmul(np.linalg.inv(self.w_H_t), w_H_ee)
            t_R_ee = t_H_ee[0:3,0:3]
            axis, angle = rot_to_axis_angle(t_R_ee, auto_align = True) 
            axis_list.append(axis)
            angle_list.append(angle)
            t_axis_angle_ee = axis * angle
            t_position_ee = t_H_ee[0:3,3]
            x_demo[k,0:3] = t_position_ee
            x_demo[k,3:6] = t_axis_angle_ee

        # solve discontinuities
        axis_list = np.array(axis_list)
        angle_list = np.array(angle_list)
        [axis, angle] = solve_discontinuities(axis_list, angle_list,
                                              flg_plot = False)
        axis_angle = matlib.repmat(angle,3,1).T * axis

        x_demo[:,3:6] = axis_angle


        if filter_mode == 'savgol':
            # compute derivatives with Savitzky-Golay filter
            x_demo = savgol_filter(x_demo, 15, 3, axis = 0, mode = 'nearest')
            dx_demo = savgol_filter(x_demo, 15, 3, axis = 0, mode = 'nearest',
                                    deriv = 1, delta = self.dt)
            ddx_demo = savgol_filter(x_demo, 15, 3, axis = 0, mode = 'nearest',
                                     deriv = 2, delta = self.dt)

        elif filter_mode == 'cd':
            dx_demo = np.zeros(x_demo.shape)
            ddx_demo = np.zeros(x_demo.shape)
            # compute derivatives with central difference
            for i in range(6):
                dx_demo[:,i] = np.gradient(x_demo[:,i],time_steps)
                ddx_demo[:,i] = np.gradient(dx_demo[:,i],time_steps)

        # set new goal
        g = x_demo[-1,:]

        # compute phase values
        s_demo = np.zeros(num_steps)
        s_demo[0] = self.canonical_sys.get_phase()
        for k in range(1,num_steps):
            dt = time_steps[k]-time_steps[k-1]
            self.canonical_sys.step(dt)
            s_demo[k] = self.canonical_sys.get_phase()
        self.canonical_sys.reset()       

        # get forcing term weights for each system dimension
        for j in range(self.sys_dim):
            # compute desired forcing term
            if self.transformation_sys.style == 'advanced':
                f_d = self.tau**2 * ddx_demo[:,j] / \
                    self.transformation_sys.K - (g[j] - x_demo[:,j]) + \
                    self.tau * self.transformation_sys.D / \
                    self.transformation_sys.K * dx_demo[:,j] + \
                    (g[j] - x_demo[0,j]) * s_demo

            elif self.transformation_sys.style == 'classic':
                if g[j]==x_demo[0,j]:
                    f_d = np.zeros(x_demo[:,j].shape)
                else:
                    f_d = (self.tau**2 * ddx_demo[:,j] - \
                        self.transformation_sys.K * (g[j] - x_demo[:,j]) + \
                        self.tau * self.transformation_sys.D * dx_demo[:,j]) / \
                        (g[j] - x_demo[0,j])  
            
            f_d = np.expand_dims(f_d, 1)

            # get regressor matrix
            Phi = np.zeros((num_steps, self.num_basis))
            for k in range(num_steps):
                basis = self.evaluate_basis(s_demo[k])[:,j] 
                Phi[k,:] = basis / np.sum(basis) * s_demo[k]

            # Solve: Phi * w = f_d
            fitted_w = np.dot(np.linalg.pinv(Phi), f_d)
            self.weights[:,j] = fitted_w[:,0]

        self.reset(x_demo[0,:], in_world_frame = False)
        self.set_new_goal(x_demo[-1,:], in_world_frame = False)

        return x_demo, dx_demo, ddx_demo, time_steps


    def get_pose(self):
        """Get end-effector pose w.r.t. world frame

        Returns
        -------
        list
            list with EE position and quaternion (in world-frame) 
        """
        # DMP state: EE pose w.r.t. target frame
        x,_ = self.get_output()

        t_position_ee = x[0:3]
        t_axis_angle_ee = x[3:6]
        t_R_ee = axis_angle_to_rot(t_axis_angle_ee)  

        # target-to-end-effector
        t_H_ee = np.zeros((4,4))
        t_H_ee[0:3,0:3] = t_R_ee
        t_H_ee[0:3,3] = t_position_ee
        t_H_ee[3,3] = 1.0

        # compose transformations
        w_H_ee = np.matmul(self.w_H_t, t_H_ee)

        # get EE frames in world coordinates
        w_R_ee = w_H_ee[0:3,0:3]

        # get position
        w_p_ee = w_H_ee[0:3,3] 

        # get orientation (as quaternion)
        w_q_ee = np.array(rot_to_quat(w_R_ee))

        ## get orientation (as axis-angle)
        # axis, angle = rot_to_axis_angle(w_R_ee, auto_align = True)
        # w_aa_ee = axis * angle

        return w_p_ee, w_q_ee


    def rollout(self):
        """Run a DMP rollout and reset.

        Returns
        -------
        list
            list of numpy.array describing the rollout
        """
        time_steps = np.arange(0., self.tau, self.dt)
        
        g0 = self.get_goal()
        s = np.zeros((len(time_steps), 1))
        x = np.zeros((len(time_steps), self.sys_dim))
        v = np.zeros((len(time_steps), self.sys_dim))
        g = np.zeros((len(time_steps), self.sys_dim))

        forcing = np.zeros((len(time_steps), self.sys_dim))

        w_p_ee = np.zeros((len(time_steps), 3))
        w_q_ee = np.zeros((len(time_steps), 4))

        for k, t in enumerate(time_steps):
            s[k] = self.get_phase()
            x[k,:], v[k,:] = self.get_output()
            g[k,:] = self.get_goal()
            forcing[k,:] = self.evaluate_forcing(s[k])
            w_p_ee[k,:], w_q_ee[k,:] = self.get_pose()
            self.step()

        self.reset(self.transformation_sys.x0, in_world_frame = False)

        return time_steps, w_p_ee, w_q_ee, x, v, g, forcing, s


    def set_new_target_frame(self, w_H_t):
        """Set new world-to-target transform

        Parameters
        ----------
        w_H_t : numpy.ndarray
            new world-to-target hom. tr. [w_H_t.shape = (4,4)]

        Raises
        ------
        ValueError
            if w_H_t is not a proper hom. tr. matrix
        """
        if w_H_t.shape != (4,4):
            if not is_rotation_matrix(w_H_t[:3,:3]):
                if not (w_H_t[3,0] == w_H_t[3,1] == w_H_t[3,2] == 0. 
                    and w_H_t[3,3] == 1): 
                    raise ValueError('w_H_t is not a proper \
                        homogeneous transformation matrix')

        self.w_H_t = w_H_t
        

    def reset(self, x0, in_world_frame = True):
        """Reset DMP at initial state x0.

        Parameters
        ----------
        x0 : numpy.ndarray
            initial state

        in_world_frame : bool, optional
            pose expressed in world frame (default is True)
        """
        if in_world_frame:
            w_position_x0 = x0[0:3]
            w_axis_angle_x0 = x0[3:6]
            w_R_x0 = axis_angle_to_rot(w_axis_angle_x0)  

            # world-to-x0
            w_H_x0 = np.zeros((4,4))
            w_H_x0[0:3,0:3] = w_R_x0
            w_H_x0[0:3,3] = w_position_x0
            w_H_x0[3,3] = 1.0

            # target-to-x0
            t_H_x0 = np.matmul(np.linalg.inv(self.w_H_t), w_H_x0)
            t_R_x0 = t_H_x0[0:3,0:3]
            axis, angle = rot_to_axis_angle(t_R_x0, auto_align = True) 
            t_axis_angle_x0 = axis*angle
            t_position_x0 = t_H_x0[0:3,3]
            x0_in_ref = np.copy(x0)
            x0_in_ref[0:3] = t_position_x0
            x0_in_ref[3:6] = t_axis_angle_x0

            # reset phase
            self.canonical_sys.reset()
             # reset transformation system
            self.transformation_sys.reset(x0_in_ref)

        else:
            # reset phase
            self.canonical_sys.reset()
            # reset transformation system
            self.transformation_sys.reset(x0)


    def set_new_goal(self, new_g, force_goal = True, in_world_frame = True):
        """Set new desired goal.

        Parameters
        ----------
        new_g : numpy.ndarray
            new goal

        force_goal : bool, optional
            choose if override g variable or only g0 (default is True)

        in_world_frame : bool, optional
            pose expressed in world frame (default is True)
        """
        if in_world_frame:
            w_position_g = new_g[0:3]
            w_axis_angle_g = new_g[3:6]
            w_R_g = axis_angle_to_rot(w_axis_angle_g)  

            # world-to-goal
            w_H_g = np.zeros((4,4))
            w_H_g[0:3,0:3] = w_R_g
            w_H_g[0:3,3] = w_position_g
            w_H_g[3,3] = 1.0

            # target-to-goal
            t_H_g = np.matmul(np.linalg.inv(self.w_H_t), w_H_g)
            t_R_g = t_H_g[0:3,0:3]
            axis, angle = rot_to_axis_angle(t_R_g, auto_align = True) 
            t_axis_angle_g = axis*angle
            t_position_g = t_H_g[0:3,3]
            g_in_ref = np.copy(new_g)
            g_in_ref[0:3] = t_position_g
            g_in_ref[3:6] = t_axis_angle_g

            # update goal system
            self.goal_sys.set_new_goal(g_in_ref, force_goal = force_goal)

        else:
             # update goal system
            self.goal_sys.set_new_goal(new_g, force_goal = force_goal)


    def get_goal(self, in_world_frame = False):
        """Set new desired goal.

        Parameters
        ----------
        in_world_frame : bool, optional
            pose expressed in world frame (default is False)
        """
        g_in_ref = DMP.get_goal(self)

        if in_world_frame:
            t_position_g = g_in_ref[0:3]
            t_axis_angle_g = g_in_ref[3:6]
            t_R_g = axis_angle_to_rot(t_axis_angle_g)  

            # target-to-goal
            t_H_g = np.zeros((4,4))
            t_H_g[0:3,0:3] = t_R_g
            t_H_g[0:3,3] = t_position_g
            t_H_g[3,3] = 1.0

            # world-to-goal
            w_H_g = np.matmul(self.w_H_t, t_H_g)
            w_R_g = w_H_g[0:3,0:3]
            axis, angle = rot_to_axis_angle(w_R_g, auto_align = True) 
            w_axis_angle_g = axis*angle
            w_position_g = w_H_g[0:3,3]
            g_in_world = np.copy(g_in_ref)
            g_in_world[0:3] = w_position_g
            g_in_world[3:6] = w_axis_angle_g

            return g_in_world

        else:
            return g_in_ref


class Bimanual_Target_DMP(Target_DMP):
    """
    A class used to implement the Target-Referred (TR) Bimanual DMP
    that controls a virtual frame placed between the two end-effectors.
    Each end-effector pose is defined by a fixed rotation and variable 
    translation from the central virtual frame. The demonstration is
    learned expressed in a target reference frame.
    Bimanual_Target_DMP extends the Target_DMP class.

    Attributes
    ----------
    canonical_sys : dmp_lib.canonical.Canonical
        canonical system

    goal_sys : dmp_lib.goal.Goal
        goal system

    transformation_sys : dmp_lib.transformation.Transformation
        transformation system

    z : numpy.ndarray
        scaled velocity variable

    x0 : numpy.ndarray
        initial state

    sys_dim: int
        system dimension

    tau : float
        time scaling parameter

    dt : float
        sampling interval

    num_basis : int
        number of basis functions

    centers: numpy.ndarray
        basis function centers

    widths: numpy.ndarray
        basis function widths

    weights: numpy.ndarray
        basis function weights

    w_H_t : numpy.ndarray
        new world-to-target hom. tr. [w_H_t.shape = (4,4)]

    R_right : numpy.ndarray
        right EE rotation matrix [R_right.shape = (3,3)]

    R_left : numpy.ndarray
        left EE rotation matrix [R_left.shape = (3,3)]
    """
    def __init__(self, num_basis, x0, g0, dt, tau,
                 alpha_phase = 25 / 3, alpha_stop = 25, alpha_g = 25 / 2,
                 K = 25**2 / 2, D = None, style = 'advanced',
                 w_H_t = np.eye(4), R_right = np.eye(3), R_left = np.eye(3)):
        """
        Parameters
        ----------
        num_basis : int
            number of basis functions

        x0 : numpy.ndarray
            initial state

        g0 : numpy.ndarray
            goal state

        dt : float
            sampling time

        tau : float
            time scaling parameter

        alpha_phase : float, optional
            phase decay parameter (default is 25 / 3)

        alpha_stop : float, optional
            phase stop parameter (default is 0)

        alpha_g : float, optional
            goal advancement parameter (default is 25 / 2)

        K : float, optional
            stiffness parameter (default is 25**2 / 2)

        D : float, optional
            damping parameter (default is None -> D = 2 * sqrt(K))

        style: string, optional
            dynamics style (default is 'advanced')

        w_H_t: numpy.ndarray, optional
            world-to-target hom. transformation (default is np.eye(4))

        R_right: numpy.ndarray, optional
            right EE rotation matrix (default is np.eye(3))

        R_left: numpy.ndarray, optional
            left EE rotation matrix (default is np.eye(3))

        Raises
        ------
        ValueError
            if w_H_t is not a proper homogeneous transformation matrix
            if R_right or R_left are not proper rotation matrices
            if len(x0) is less than 7 (virtual frame prose + distance)
        """
        if x0.size < 7:
            raise ValueError('len(x0) must be at least 7 \
                (virtual frame prose + distance)')

        if not is_rotation_matrix(R_right):
            raise ValueError('R_right is not a rotation matrix')

        if not is_rotation_matrix(R_left):
            raise ValueError('R_left is not a rotation matrix')

        Target_DMP.__init__(self,num_basis, x0, g0, dt, tau,
            alpha_phase, alpha_stop, alpha_g, K, D, style, w_H_t)

        self.R_right = R_right
        self.R_left = R_left

    def get_poses(self):
        """Get end-effector positions from virtual frame pose.

        Returns
        -------
        list
            list with left and right positions and quaternions 
        """
        # DMP state: virtual frame pose & distance (+ others)
        x,_ = self.get_output()

        # virtual frame pose in world frame
        vf_position, vf_quat = self.get_pose()

        vf_R = quat_to_rot(vf_quat)
        dist = x[6]

        # world-to-virtual-frame
        H_vf = np.zeros((4,4))
        H_vf[0:3,0:3] = vf_R
        H_vf[0:3,3] = vf_position
        H_vf[3,3] = 1.0

        # virtual-frame-to-right
        vf_H_r = np.zeros((4,4))
        vf_H_r[0:3,0:3] = self.R_right
        vf_H_r[0:3,3] = np.array([0, 0, - dist / 2])
        vf_H_r[3,3] = 1.0

        # virtual-frame-to-left
        vf_H_l = np.zeros((4,4))
        vf_H_l[0:3,0:3] = self.R_left
        vf_H_l[0:3,3] = np.array([0, 0, dist / 2])
        vf_H_l[3,3] = 1.0

        # compose transformations
        H_r = np.matmul(H_vf, vf_H_r)
        H_l = np.matmul(H_vf, vf_H_l)

        R_r = H_r[0:3,0:3]
        q_r = np.array(rot_to_quat(R_r))
        p_r = H_r[0:3,3]

        R_l = H_l[0:3,0:3]
        q_l = np.array(rot_to_quat(R_l))
        p_l = H_l[0:3,3]

        return p_r, q_r, p_l, q_l

    def rollout(self):
        """Run a DMP rollout and reset.

        Returns
        -------
        list
            list of numpy.array describing the bimanual rollout
        """
        time_steps = np.arange(0., self.tau, self.dt)
        
        g0 = self.get_goal(in_world_frame = False)
        s = np.zeros((len(time_steps), 1))
        x = np.zeros((len(time_steps), self.sys_dim))
        v = np.zeros((len(time_steps), self.sys_dim))
        g = np.zeros((len(time_steps), self.sys_dim))

        forcing = np.zeros((len(time_steps), self.sys_dim))

        p_r = np.zeros((len(time_steps), 3))
        q_r = np.zeros((len(time_steps), 4))
        p_l = np.zeros((len(time_steps), 3))
        q_l = np.zeros((len(time_steps), 4))

        for k, t in enumerate(time_steps):
            s[k] = self.get_phase()
            x[k,:], v[k,:] = self.get_output()
            g[k,:] = self.get_goal()
            forcing[k,:] = self.evaluate_forcing(s[k])
            p_r[k,:], q_r[k,:], p_l[k,:], q_l[k,:] = self.get_poses()
            self.step()

        self.reset(self.transformation_sys.x0)

        return time_steps, p_r, q_r, p_l, q_l, x, v, g, forcing, s

    def info(self):
        """Return info about DMP type.

        Returns
        -------
        string
            DMP type
        """
        return 'dual'
