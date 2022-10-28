import numpy as np
import matplotlib.pyplot as plt
from dmp_lib.movement_primitives import DMP
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from dmp_lib.geometry import axis_angle_to_rot as axis_angle_to_rot
from dmp_lib.geometry import rot_to_axis_angle as rot_to_axis_angle
from dmp_lib.geometry import quat_to_rot as quat_to_rot
from dmp_lib.geometry import quat_to_axis_angle as quat_to_axis_angle

demo = np.loadtxt('DATA/single_hand_pick_box.csv', delimiter=',')

time_steps = demo[:,0] # 1st col: time
time_steps = time_steps - time_steps[0] 
x_demo = demo[:,1:] # 2nd-4th col: position / 5th-8th col: axis-angle

dt = 0.01

tau = time_steps[-1]
x0 = x_demo[0,:]
g0 = x_demo[-1,:]

N = 100

mp = DMP(num_basis = N, tau = tau, dt = dt, x0 = x0, g0 = g0)
x_demo, dx_demo, ddx_demo, time_steps = mp.learn_from_demo(x_demo, time_steps)
time_steps, x, v, g, forcing, s = mp.rollout()


fig, axs = plt.subplots(nrows = 2, ncols = 2)
axs[0,0].plot(time_steps,x[:,0:3], linewidth=2)
axs[0,0].plot(time_steps,x_demo[:,0:3], 'k--')
axs[0,0].grid()
axs[0,0].set_title('Pose', fontstyle = 'italic')
axs[0,0].set(xlabel='Time [s]', ylabel = r'$\mathbf{x}$ [m]')
axs[1,0].plot(time_steps,x[:,3:6], linewidth=2)
axs[1,0].plot(time_steps,x_demo[:,3:6], 'k--')
axs[1,0].grid()
axs[1,0].set(xlabel='Time [s]', ylabel = r'$\mathbf{\theta}$ [rad]')
axs[0,1].plot(time_steps,v[:,0:3], linewidth=2)
axs[0,1].plot(time_steps,dx_demo[:,0:3], 'k--')
axs[0,1].grid()
axs[0,1].set_title('Velocities', fontstyle='italic')
axs[0,1].set(xlabel='Time [s]', ylabel = r'$\dot{\mathbf{x}}$ [m/s]')
axs[1,1].plot(time_steps,v[:,3:6], linewidth=2)
axs[1,1].plot(time_steps,dx_demo[:,3:6], 'k--')
axs[1,1].grid()
axs[1,1].set(xlabel='Time [s]', ylabel = r'$\dot{\mathbf{\theta}}$ [rad/s]')
fig.tight_layout()
# plt.show()


fig = plt.figure()
ax = plt.axes(projection = '3d')
x_max = np.max(x[:,0])
y_max = np.max(x[:,1])
z_max = np.max(x[:,2])
x_min = np.min(x[:,0])
y_min = np.min(x[:,1])
z_min = np.min(x[:,2])
scale = np.max((x_max - x_min, y_max - y_min, z_max - z_min))
x_avg = (x_min + x_max) / 2
y_avg = (y_min + y_max) / 2
z_avg = (z_min + z_max) / 2

# plot frames
for k in range(0,x_demo.shape[0], 50):
    R = axis_angle_to_rot(x[k,3:6])
    ax.quiver(x[k,0], x[k,1], x[k,2], R[0,0], R[1,0], R[2,0], colors = 'r', 
                length = 0.05, normalize = True)
    ax.quiver(x[k,0], x[k,1], x[k,2], R[0,1], R[1,1], R[2,1], colors = 'g', 
                length = 0.05, normalize = True)
    ax.quiver(x[k,0], x[k,1], x[k,2], R[0,2], R[1,2], R[2,2], colors = 'b', 
                length = 0.05, normalize = True)

# plot trajectories
ax.plot3D(x[:,0], x[:,1], x[:,2], 'k', linewidth = 2,  label = 'right pose')
ax.set_xlim3d([x_avg - scale / 2, x_avg + scale / 2])
ax.set_ylim3d([y_avg - scale / 2, y_avg + scale / 2])
ax.set_zlim3d([z_avg - scale / 2, z_avg + scale / 2])
ax.set_aspect('equal', adjustable = 'box')
plt.show()

