import numpy as np
import numpy.matlib as matlib
from dmp_lib.transformation import Transformation
from dmp_lib.canonical import Canonical
from dmp_lib.goal import Goal
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
from scipy.signal import savgol_filter
from dmp_lib.geometry import quat_to_rot
from dmp_lib.geometry import rot_to_quat
from dmp_lib.geometry import rot_to_axis_angle
from dmp_lib.geometry import axis_angle_to_rot
from dmp_lib.geometry import axis_angle_to_quat
from dmp_lib.geometry import quat_to_axis_angle
from dmp_lib.geometry import solve_discontinuities



class DMP():
    """ Implementation of Dynamic Movement Primitive (DMP) """
    def __init__(self, num_basis, x0, g0, dt, tau,
                 alpha_phase = 25./3, alpha_g = 25./2, K = 25.**2/2, D = None, alpha_stop = 25., style = 'advanced'):
        """
        num_basis int       : number of basis functions
        x0 array            : initial state [sys_dim]
        g array             : goal state [sys_dim]
        dt float            : sampling time
        tau float           : time constant (= final time)
        alpha_phase float   : canonical system parameter
        alpha_g float       : goal system parameter
        K float             : elastic parameter in the dynamical system
        D float             : damping parameter in the dynamical system [by default, D = 2*sqrt(K)]
        alpha_stop          : phase stop parameter
        style               : 'advanced' or 'classic' formulation
        """
        if np.isscalar(x0):
            x0 = np.array([x0])
        if np.isscalar(g0):
            g0 = np.array([g0])

        self.sys_dim = x0.size # save system dimension

        self.canonical_sys = Canonical(tau, alpha_phase, alpha_stop) # init canonical system
        self.goal_sys = Goal(g0, tau, alpha_g) # init goal system
        self.transformation_sys = Transformation(x0, tau, K, D, style) # init transformation system
        self.tau = tau
        self.dt = dt
        self.num_basis = num_basis
        self.init_forcing(num_basis) # init forcing


    def init_forcing(self, num_basis):
        """
        Initialize parameters of the forcing term
        num_basis int       : number of basis functions
        """
        self.centers = np.zeros((num_basis, self.sys_dim))
        self.widths = np.zeros((num_basis, self .sys_dim))
        self.weights = np.zeros((num_basis, self.sys_dim))
        # compute Gaussian centers
        c_locations = np.exp(-self.canonical_sys.alpha_phase * np.array(list(range(num_basis)))/(num_basis-1))
        self.centers = matlib.repmat(np.expand_dims(c_locations,1), 1, self.sys_dim) 
        # compute Gaussian widths
        for k in range(num_basis-1):
            self.widths[k,:] = 1/(self.centers[k+1,:]-self.centers[k,:])**2 
        self.widths[-1,:] = self.widths[-2,:]


    def evaluate_basis(self, s):
        """
        Evaluate basis functions in s
        s float         : phase location
        """
        if np.size(s) == 1:
            return np.exp(-self.widths*(s-self.centers)**2)


    def evaluate_forcing(self, s):
        """
        Evaluate forcing term in s
        s float         : phase location
        """
        all_phi_s = self.evaluate_basis(s)
        den = np.sum(all_phi_s, 0)
        if np.sum(den) > 0.:
            f = np.sum(all_phi_s*self.weights,0)/den*s
        else:
            f = np.zeros(self.sys_dim)
        return f


    def evaluate_weighted_basis(self, s):
        """
        Evaluate weighted basis functions in s
        s float         : phase location
        """
        all_phi_s = self.evaluate_basis(s)
        return all_phi_s*self.weights


    def reset(self, x0):
        """
        Reset DMP in initial state x0
        x0 array        : initial state
        """
        if np.isscalar(x0):
            x0 = np.array([x0])

        self.canonical_sys.reset() # reset phase
        self.transformation_sys.reset(x0) # reset transformation system


    def set_tau(self, new_tau):
        self.tau = new_tau
        self.canonical_sys.tau = new_tau
        self.transformation_sys.tau = new_tau
        self.goal_sys.tau = new_tau


    def set_new_goal(self, new_g, force_goal=True):
        """
        Set new goal
        new_g array     : new goal 
        force_goal      : override g variable (or only g0)
        """
        self.goal_sys.set_new_goal(new_g, force_goal=force_goal) # update goal system


    def get_phase(self):
        """
        Get current phase from canonical system
        """
        return self.canonical_sys.get_phase()


    def get_goal(self):
        """
        Get current goal from goal system
        """
        return self.goal_sys.get_goal()


    def get_output(self):
        """
        Get DMP output from transformation system
        """
        [x, z] = self.transformation_sys.get_state()
        return x, z/self.tau


    def step(self, dist=0):
        """
        Single-step DMP update
        """
        s = self.get_phase()
        g = self.get_goal()
        f = self.evaluate_forcing(s)

        self.transformation_sys.step(self.dt, g, s, f)
        self.canonical_sys.step(self.dt, dist=dist)
        self.goal_sys.step(self.dt)


    def learn_from_demo(self, x_demo, time_steps, filter_mode = 'cd'):
        """
        Fit forcing function to reproduce desired trajectory (tau, x0, g0 are changed according to demo)
        x_demo array        : target [num_steps x sys_dim]
        time_steps array    : time steps [num_steps]
        filter_mode string  : select filter between Savitzky-Golay ('savgol') or central difference ('cd')
        """
        x_demo = np.reshape(x_demo, (-1, self.sys_dim))
        time_steps = time_steps-time_steps[0] # initial time to 0 sec

        if x_demo.shape[0] != time_steps.shape[0]:
            raise ValueError("x_demo and time_steps must have the same number of samples")

        path_gen = interpolate.interp1d(time_steps, x_demo.T) # interpolate demo
        time_steps = np.arange(0., time_steps[-1], self.dt) # use constant sampling intervals
                
        num_steps = len(time_steps)

        # self.set_tau(time_steps[-1]) # set tau equal to demo time length
        self.set_tau(num_steps*self.dt)

        x_demo = path_gen(time_steps).T

        if filter_mode == 'savgol':
            # compute derivatives with Savitzky-Golay filter
            x_demo = savgol_filter(x_demo, 15, 3, axis=0, mode='nearest')
            dx_demo = savgol_filter(x_demo, 15, 3, axis=0, mode='nearest', deriv=1, delta=self.dt)
            ddx_demo = savgol_filter(x_demo, 15, 3, axis=0, mode='nearest', deriv=2, delta=self.dt)

        elif filter_mode == 'cd':
            dx_demo = np.zeros(x_demo.shape)
            ddx_demo = np.zeros(x_demo.shape)
            # compute derivatives with central difference
            for i in range(self.sys_dim):
                dx_demo[:,i] = np.gradient(x_demo[:,i],time_steps)
                ddx_demo[:,i] = np.gradient(dx_demo[:,i],time_steps)

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
        f_d_all = np.zeros((self.sys_dim, num_steps))
        for j in range(self.sys_dim):
            # compute desired forcing term
            if self.transformation_sys.style == 'advanced':
                f_d = self.tau**2*ddx_demo[:,j]/self.transformation_sys.K-(g[j]-x_demo[:,j])+self.tau*self.transformation_sys.D/self.transformation_sys.K*dx_demo[:,j]+(g[j]-x_demo[0,j])*s_demo            
            elif self.transformation_sys.style == 'classic':
                if g[j]==x_demo[0,j]:
                    f_d = np.zeros(x_demo[:,j].shape)
                else:
                    f_d = (self.tau**2*ddx_demo[:,j]-self.transformation_sys.K*(g[j]-x_demo[:,j])+self.tau*self.transformation_sys.D*dx_demo[:,j])/(g[j]-x_demo[0,j])  
            f_d_all[j,:] = f_d

        for j in range(self.sys_dim):
            f_d = f_d_all[j:j+1,:].T
            # get regressor matrix
            Phi = np.zeros((num_steps, self.num_basis))
            for k in range(num_steps):
                basis = self.evaluate_basis(s_demo[k])[:,j] 
                Phi[k,:] = basis/np.sum(basis)*s_demo[k]

            # Solve: Phi * w = f_d
            fitted_w = np.dot(np.linalg.pinv(Phi), f_d)
            self.weights[:,j] = fitted_w[:,0]


        return x_demo, dx_demo, ddx_demo, time_steps


    def get_weights(self):
        """
        Get DMP weights
        """
        return self.weights


    def import_weights(self, new_weights):
        """
        Import DMP weights
        new_weights array   : new DMP weights [num_basis x sys_dim]
        """
        if new_weights.shape == self.weights.shape:
            self.weights = new_weights
        else:
            print("Weights dimensions not valid")

    def rollout(self):
        """
        Run a DMP rollout and reset (for visualization purposes)
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
        """
        Is the DMP single or dual?
        """
        return 'single'


class Bimanual_DMP(DMP):
    """ Implementation of Bimanual Dynamic Movement Primitive (BiDMP) 
        - BiDMP controls a virtual frame placed between the two end-effectors
        - each end-effector pose is defined by a fixed rotation and variable translation from the virtual frame
    """
    def __init__(self, num_basis, x0, g0, dt, tau,
                 alpha_phase = 25./3, alpha_g = 25./2, K = 25.**2/2, D = None, alpha_stop = 25.,
                 style = 'advanced', R_right = np.eye(3), R_left = np.eye(3)):
        """
        num_basis int       : number of basis functions
        x0 array            : initial state [sys_dim]
        g array             : goal state [sys_dim]
        dt float            : sampling time
        tau float           : time constant (= final time)
        alpha_phase float   : canonical system parameter
        alpha_g float       : goal system parameter
        K float             : elastic parameter in the dynamical system
        D float             : damping parameter in the dynamical system [by default, D = 2*sqrt(K)]
        alpha_stop          : phase stop parameter
        style               : 'advanced' or 'classic' formulation
        R_right             : right EE rotation matrix [3x3]
        R_left              : left EE rotation matrix [3x3]
        """
        if np.isscalar(x0):
            x0 = np.array([x0])

        if x0.size < 7:
            raise ValueError('x0 must have at least dimensions 7 (virtual frame prose + distance)')

        if R_right.shape != (3,3) and np.linalg.abs(np.linalg.det(R_right)-1) > 10**-3:
            raise ValueError('R_right is not a rotation matrix')

        if R_left.shape != (3,3) and np.linalg.abs(np.linalg.det(R_left)-1) > 10**-3:
            raise ValueError('R_left is not a rotation matrix')

        DMP.__init__(self,num_basis, x0, g0, dt, tau, alpha_phase, alpha_g, K, D, alpha_stop, style)

        self.R_right = R_right
        self.R_left = R_left

    def get_poses(self):
        """
        Get end-effector positions from virtual frame pose
        """
        x,_ = self.get_output() # DMP state: virtual frame pose & distance (+ other components not used here)

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
        """
        Run a DMP rollout and reset (for visualization purposes)
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
        """
        Is the DMP single or dual?
        """
        return 'dual'



class Target_DMP(DMP):
    """ Implementation of Target-Based Dynamic Movement Primitive 
        - Target-Based DMP defines the trajectory w.r.t. a target frame 
    """
    def __init__(self, num_basis, x0, g0, dt, tau,
                 alpha_phase = 25./3, alpha_g = 25./2, K = 25.**2/2, D = None, alpha_stop = 25.,
                 style = 'advanced', w_H_t = np.eye(4)):
        """
        num_basis int       : number of basis functions
        x0 array            : initial state (in target frame) [sys_dim]
        g array             : goal state (in target frame) [sys_dim]
        dt float            : sampling time
        tau float           : time constant (= final time)
        alpha_phase float   : canonical system parameter
        alpha_g float       : goal system parameter
        K float             : elastic parameter in the dynamical system
        D float             : damping parameter in the dynamical system [by default, D = 2*sqrt(K)]
        alpha_stop          : phase stop parameter
        style               : 'advanced' or 'classic' formulation
        w_H_t               : world-to-target homogeneous transformation [4x4]
        """
        if np.isscalar(x0):
            x0 = np.array([x0])

        if x0.size < 6:
            raise ValueError('x0 must represent a 6D end-effector pose')

        DMP.__init__(self,num_basis, x0, g0, dt, tau, alpha_phase, alpha_g, K, D, alpha_stop, style)

        if w_H_t.shape != (4,4):
            if np.linalg.abs(np.linalg.det(R_right)-1) > 10**-3:
                if not (w_H_t[3,0] == w_H_t[3,1] == w_H_t[3,2] == 0. and w_H_t[3,3] == 1): 
                    raise ValueError('w_H_t is not a proper homogeneous transformation matrix')

        self.w_H_t = w_H_t

    def learn_from_demo(self, x_demo, time_steps, filter_mode = 'cd'):
        """
        Fit forcing function to reproduce desired trajectory (tau, x0, g0 are changed according to demo)
        x_demo array        : target [num_steps x sys_dim]
        time_steps array    : time steps [num_steps]
        filter_mode string  : select filter between Savitzky-Golay ('savgol') or central difference ('cd')
        """
        x_demo = np.reshape(x_demo, (-1, self.sys_dim))
        time_steps = time_steps-time_steps[0] # initial time to 0 sec

        path_gen = interpolate.interp1d(time_steps, x_demo.T) # interpolate demo
        time_steps = np.arange(0., time_steps[-1], self.dt) # use constant sampling intervals
        
        num_steps = len(time_steps)

        # self.set_tau(time_steps[-1]) # set tau equal to demo time length
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
            axis, angle = rot_to_axis_angle(t_R_ee, auto_align=True) 
            axis_list.append(axis)
            angle_list.append(angle)
            t_axis_angle_ee = axis*angle
            t_position_ee = t_H_ee[0:3,3]
            x_demo[k,0:3] = t_position_ee
            x_demo[k,3:6] = t_axis_angle_ee

        # solve discontinuities
        axis_list = np.array(axis_list)
        angle_list = np.array(angle_list)
        [axis, angle] = solve_discontinuities(axis_list, angle_list, flg_plot=False)
        axis_angle = matlib.repmat(angle,3,1).T*axis

        x_demo[:,3:6] = axis_angle


        if filter_mode == 'savgol':
            # compute derivatives with Savitzky-Golay filter
            x_demo = savgol_filter(x_demo, 15, 3, axis=0, mode='nearest')
            dx_demo = savgol_filter(x_demo, 15, 3, axis=0, mode='nearest', deriv=1, delta=self.dt)
            ddx_demo = savgol_filter(x_demo, 15, 3, axis=0, mode='nearest', deriv=2, delta=self.dt)

        elif filter_mode == 'cd':
            dx_demo = np.zeros(x_demo.shape)
            ddx_demo = np.zeros(x_demo.shape)
            # compute derivatives with central difference
            for i in range(6):
                dx_demo[:,i] = np.gradient(x_demo[:,i],time_steps)
                ddx_demo[:,i] = np.gradient(dx_demo[:,i],time_steps)

        # self.reset(x_demo[0,:]) # reset DMP to x0 = x_demo[0]
        # self.set_new_goal(x_demo[-1,:]) # set new goal
        # g = self.get_goal()
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
                f_d = self.tau**2*ddx_demo[:,j]/self.transformation_sys.K-(g[j]-x_demo[:,j])+self.tau*self.transformation_sys.D/self.transformation_sys.K*dx_demo[:,j]+(g[j]-x_demo[0,j])*s_demo            
            elif self.transformation_sys.style == 'classic':
                if g[j]==x_demo[0,j]:
                    f_d = np.zeros(x_demo[:,j].shape)
                else:
                    f_d = (self.tau**2*ddx_demo[:,j]-self.transformation_sys.K*(g[j]-x_demo[:,j])+self.tau*self.transformation_sys.D*dx_demo[:,j])/(g[j]-x_demo[0,j])  
            
            f_d = np.expand_dims(f_d, 1)

            # get regressor matrix
            Phi = np.zeros((num_steps, self.num_basis))
            for k in range(num_steps):
                basis = self.evaluate_basis(s_demo[k])[:,j] 
                Phi[k,:] = basis/np.sum(basis)*s_demo[k]

            # Solve: Phi * w = f_d
            fitted_w = np.dot(np.linalg.pinv(Phi), f_d)
            self.weights[:,j] = fitted_w[:,0]

        self.reset(x_demo[0,:], in_world_frame=False)
        self.set_new_goal(x_demo[-1,:], in_world_frame=False)

        return x_demo, dx_demo, ddx_demo, time_steps

    def get_pose(self):
        """
        Get end-effector pose w.r.t. world frame
        """
        x,_ = self.get_output() # DMP state: EE pose w.r.t. target frame

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
        w_q_ee = np.array(rot_to_quat(w_R_ee)) # orientation (as quaternion)
        # axis, angle = rot_to_axis_angle(w_R_ee, auto_align=True)
        # w_aa_ee = axis*angle # orientation (as axis-angle)
        w_p_ee = w_H_ee[0:3,3] # position

        return w_p_ee, w_q_ee

    def rollout(self):
        """
        Run a DMP rollout and reset (for visualization purposes)
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

        self.reset(self.transformation_sys.x0, in_world_frame=False)

        return time_steps, w_p_ee, w_q_ee, x, v, g, forcing, s


    def set_new_target_frame(self,w_H_t):
        """
        Set new world-to-target transform
        w_H_t                             : world-to-target homogeneous transformation [4x4]
        """
        if w_H_t.shape != (4,4):
            if np.linalg.abs(np.linalg.det(R_right)-1) > 10**-3:
                if not (w_H_t[3,0] == w_H_t[3,1] == w_H_t[3,2] == 0. and w_H_t[3,3] == 1): 
                    raise ValueError('w_H_t is not a proper homogeneous transformation matrix')

        self.w_H_t = w_H_t
        

    def reset(self, x0, in_world_frame = True):
        """
        Reset DMP in initial state x0
        x0 array        : initial state
        in_world_frame  : pose expressed in world frame
        """
        if np.isscalar(x0):
            x0 = np.array([x0])

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
            axis, angle = rot_to_axis_angle(t_R_x0, auto_align=True) 
            t_axis_angle_x0 = axis*angle
            t_position_x0 = t_H_x0[0:3,3]
            x0_in_ref = np.copy(x0)
            x0_in_ref[0:3] = t_position_x0
            x0_in_ref[3:6] = t_axis_angle_x0

            self.canonical_sys.reset() # reset phase
            self.transformation_sys.reset(x0_in_ref) # reset transformation system

        else:

            self.canonical_sys.reset() # reset phase
            self.transformation_sys.reset(x0) # reset transformation system


    def set_new_goal(self, new_g, force_goal=True, in_world_frame=True):
        """
        Set new goal
        new_g array     : new goal 
        force_goal      : override g variable (or only g0)
        in_world_frame  : pose expressed in world frame
        """
        if np.isscalar(new_g):
            new_g = np.array([new_g])

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
            axis, angle = rot_to_axis_angle(t_R_g, auto_align=True) 
            t_axis_angle_g = axis*angle
            t_position_g = t_H_g[0:3,3]
            g_in_ref = np.copy(new_g)
            g_in_ref[0:3] = t_position_g
            g_in_ref[3:6] = t_axis_angle_g

            self.goal_sys.set_new_goal(g_in_ref, force_goal=force_goal) # update goal system

        else:

            self.goal_sys.set_new_goal(new_g, force_goal=force_goal) # update goal system


    def get_goal(self, in_world_frame=False):
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
            axis, angle = rot_to_axis_angle(w_R_g, auto_align=True) 
            w_axis_angle_g = axis*angle
            w_position_g = w_H_g[0:3,3]
            g_in_world = np.copy(g_in_ref)
            g_in_world[0:3] = w_position_g
            g_in_world[3:6] = w_axis_angle_g

            return g_in_world


        else:
            return g_in_ref


class Bimanual_Target_DMP(Target_DMP):
    """ Implementation of Target-Referred Bimanual Dynamic Movement Primitive (BiDMP) 
        - BiDMP controls a virtual frame placed between the two end-effectors
        - each end-effector pose is defined by a fixed rotation and variable translation from the virtual frame
    """
    def __init__(self, num_basis, x0, g0, dt, tau,
                 alpha_phase = 25./3, alpha_g = 25./2, K = 25.**2/2, D = None, alpha_stop = 25.,
                 style = 'advanced', w_H_t = np.eye(4), R_right = np.eye(3), R_left = np.eye(3)):
        """
        num_basis int       : number of basis functions
        x0 array            : initial state [sys_dim]
        g array             : goal state [sys_dim]
        dt float            : sampling time
        tau float           : time constant (= final time)
        alpha_phase float   : canonical system parameter
        alpha_g float       : goal system parameter
        K float             : elastic parameter in the dynamical system
        D float             : damping parameter in the dynamical system [by default, D = 2*sqrt(K)]
        alpha_stop          : phase stop parameter
        style               : 'advanced' or 'classic' formulation
        w_H_t               : world-to-target reference frame (for Target-Referred DMP)
        R_right             : right EE rotation matrix [3x3]
        R_left              : left EE rotation matrix [3x3]
        """
        if np.isscalar(x0):
            x0 = np.array([x0])

        if x0.size < 7:
            raise ValueError('x0 must have at least dimensions 7 (virtual frame prose + distance)')

        if R_right.shape != (3,3) and np.linalg.abs(np.linalg.det(R_right)-1) > 10**-3:
            raise ValueError('R_right is not a rotation matrix')

        if R_left.shape != (3,3) and np.linalg.abs(np.linalg.det(R_left)-1) > 10**-3:
            raise ValueError('R_left is not a rotation matrix')

        Target_DMP.__init__(self,num_basis, x0, g0, dt, tau, alpha_phase, alpha_g, K, D, alpha_stop, style, w_H_t)

        self.R_right = R_right
        self.R_left = R_left

    def get_poses(self):
        """
        Get end-effector positions from virtual frame pose
        """
        x,_ = self.get_output() # DMP state: virtual frame pose & distance (+ other components not used here)
        vf_position, vf_quat = self.get_pose() # virtual frame pose in world frame

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
        """
        Run a DMP rollout and reset (for visualization purposes)
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
        """
        Is the DMP single or dual?
        """
        return 'dual'
