#!/usr/bin/env python

# Copyright 2022 by Fabio Amadio.
# All rights reserved.
# This file is part of the dmp_lib,
# and is released under the "GNU General Public License".
# Please see the LICENSE file included in the package.

import numpy as np
import matplotlib.pyplot as plt
from dmp_lib.movement_primitives import Target_DMP
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from dmp_lib.geometry import axis_angle_to_rot as axis_angle_to_rot
from dmp_lib.geometry import rot_to_axis_angle as rot_to_axis_angle
from dmp_lib.geometry import quat_to_rot as quat_to_rot
from dmp_lib.geometry import quat_to_axis_angle as quat_to_axis_angle

demo = np.loadtxt('DATA/demo_rectangle2.csv', delimiter=',')

time_steps = demo[:,0] # 1st col: time
time_steps = time_steps - time_steps[0] 
x_demo = demo[:,1:] # 2nd-4th col: position / 5th-8th col: axis-angle

dt = 0.01

tau = time_steps[-1]
x0 = x_demo[0,:]
g0 = x_demo[-1,:]

w_R_t = axis_angle_to_rot(g0[3:])

w_H_t = np.zeros((4,4))
w_H_t[0:3,0:3] = w_R_t
w_H_t[0:3,3] = g0[:3]
w_H_t[3,3] = 1.0

N = 100

mp = Target_DMP(num_basis = N, tau = tau, dt = dt, x0 = x0, g0 = g0, style = 'advanced', w_H_t = w_H_t)
mp.learn_from_demo(x_demo, time_steps) # it set x0 and g0 according to w_H_t
mp.reset(x0+[0,0,-400,0,0,0], in_world_frame = True)
time_steps, w_p_ee, w_q_ee, x, v, g, forcing, s = mp.rollout()

fig = plt.figure()
ax = plt.axes(projection = '3d')
x_max = np.max(w_p_ee[:,0])
y_max = np.max(w_p_ee[:,1])
z_max = np.max(w_p_ee[:,2])
x_min = np.min(w_p_ee[:,0])
y_min = np.min(w_p_ee[:,1])
z_min = np.min(w_p_ee[:,2])
scale = np.max((x_max - x_min, y_max - y_min, z_max - z_min))
x_avg = (x_min + x_max) / 2
y_avg = (y_min + y_max) / 2
z_avg = (z_min + z_max) / 2

# plot frames
for k in range(0,x_demo.shape[0], 50):
    R = quat_to_rot(w_q_ee[k,:])
    ax.quiver(w_p_ee[k,0], w_p_ee[k,1], w_p_ee[k,2], R[0,0], R[1,0], R[2,0], colors = 'r', 
                length = 10, normalize = True)
    ax.quiver(w_p_ee[k,0], w_p_ee[k,1], w_p_ee[k,2], R[0,1], R[1,1], R[2,1], colors = 'g', 
                length = 10, normalize = True)
    ax.quiver(w_p_ee[k,0], w_p_ee[k,1], w_p_ee[k,2], R[0,2], R[1,2], R[2,2], colors = 'b', 
                length = 10, normalize = True)

# plot trajectories
ax.plot3D(w_p_ee[:,0], w_p_ee[:,1], w_p_ee[:,2], 'k', linewidth = 2,  label = 'right pose')
ax.set_xlim3d([x_avg - scale / 2, x_avg + scale / 2])
ax.set_ylim3d([y_avg - scale / 2, y_avg + scale / 2])
ax.set_zlim3d([z_avg - scale / 2, z_avg + scale / 2])
ax.set_aspect('equal', adjustable = 'box')
plt.show()

