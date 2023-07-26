#!/usr/bin/env python

# Copyright 2022 by Fabio Amadio.
# All rights reserved.
# This file is part of the dmp_lib,
# and is released under the "GNU General Public License".
# Please see the LICENSE file included in the package.

import numpy as np
import matplotlib.pyplot as plt
from dmp_lib.movement_primitives import DMP
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from dmp_lib.geometry import axis_angle_to_rot as axis_angle_to_rot
from dmp_lib.geometry import rot_to_axis_angle as rot_to_axis_angle
from dmp_lib.geometry import quat_to_rot as quat_to_rot
from dmp_lib.geometry import quat_to_axis_angle as quat_to_axis_angle

demo = np.loadtxt('DATA/demo_rectangle.csv', delimiter=',')

time_steps = demo[:,0] # 1st col: time
time_steps = time_steps - time_steps[0] 
x_demo = demo[:,1:] # 2nd-4th col: position / 5th-8th col: axis-angle

dt = 0.01

tau = time_steps[-1]
x0 = x_demo[0,:]
g0 = x_demo[-1,:]

N = 100

mp_none = DMP(num_basis = N, tau = tau, dt = dt, x0 = x0, g0 = g0)
mp_diag = DMP(num_basis = N, tau = tau, dt = dt, x0 = x0, g0 = g0, rescale = 'diagonal')
mp_roto = DMP(num_basis = N, tau = tau, dt = dt, x0 = x0, g0 = g0, rescale = 'rotodilation')

mp_none.learn_from_demo(x_demo, time_steps)
mp_diag.learn_from_demo(x_demo, time_steps)
mp_roto.learn_from_demo(x_demo, time_steps)

# Change initial state
x0 = x0 + [0,0,-0.4,0,0,0]
mp_none.reset(x0)
mp_diag.reset(x0)
mp_roto.reset(x0)

time_steps, x_none, _, _, _, _ = mp_none.rollout()
_, x_diag, _, _, _, _ = mp_diag.rollout()
_, x_roto, _, _, _, _ = mp_roto.rollout()


fig = plt.figure()
ax = plt.axes(projection = '3d')
x_max = np.max(x_none[:,0])
y_max = np.max(x_none[:,1])
z_max = np.max(x_none[:,2])
x_min = np.min(x_none[:,0])
y_min = np.min(x_none[:,1])
z_min = np.min(x_none[:,2])
scale = np.max((x_max - x_min, y_max - y_min, z_max - z_min))
x_avg = (x_min + x_max) / 2
y_avg = (y_min + y_max) / 2
z_avg = (z_min + z_max) / 2


# plot trajectories
ax.plot3D(x_none[:,0], x_none[:,1], x_none[:,2], 'k', linewidth = 2,  label = 'No rescaling')
ax.plot3D(x_diag[:,0], x_diag[:,1], x_diag[:,2], 'r', linewidth = 2,  label = 'Diagonal')
ax.plot3D(x_roto[:,0], x_roto[:,1], x_roto[:,2], 'b', linewidth = 2,  label = 'Rotodilation')
ax.set_xlim3d([x_avg - scale / 2, x_avg + scale / 2])
ax.set_ylim3d([y_avg - scale / 2, y_avg + scale / 2])
ax.set_zlim3d([z_avg - scale / 2, z_avg + scale / 2])
ax.set_aspect('equal', adjustable = 'box')
ax.legend()
plt.show()

