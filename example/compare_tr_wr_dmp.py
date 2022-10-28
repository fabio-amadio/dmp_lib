import numpy as np
import numpy.matlib as matlib
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import argparse
import yaml
import pickle
from dmp_lib.movement_primitives import Bimanual_DMP
from dmp_lib.movement_primitives import Bimanual_Target_DMP
from dmp_lib.geometry import rot_to_quat
from dmp_lib.geometry import quat_to_rot
from dmp_lib.geometry import rot_to_axis_angle
from dmp_lib.geometry import axis_angle_to_rot
from dmp_lib.geometry import axis_angle_to_quat
from dmp_lib.geometry import quat_to_axis_angle
from dmp_lib.geometry import solve_discontinuities
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D


def change_frame(trajectory, H_target):
    # express in target reference frame
    axis_list = []
    angle_list = []
    for k in range(trajectory.shape[0]):       

        w_position_ee = trajectory[k,0:3]
        w_quat_ee = trajectory[k,3:7]
        w_R_ee = quat_to_rot(w_quat_ee)  

        # world-to-end-effector
        w_H_ee = np.zeros((4,4))
        w_H_ee[0:3,0:3] = w_R_ee
        w_H_ee[0:3,3] = w_position_ee
        w_H_ee[3,3] = 1.0

        # target-to-end-effector
        t_H_ee = np.matmul(np.linalg.inv(H_target), w_H_ee)
        t_R_ee = t_H_ee[0:3,0:3]
        axis, angle = rot_to_axis_angle(t_R_ee, auto_align=True) 
        axis_list.append(axis)
        angle_list.append(angle)
        t_axis_angle_ee = axis*angle
        t_position_ee = t_H_ee[0:3,3]
        trajectory[k,0:3] = t_position_ee
        trajectory[k,3:6] = t_axis_angle_ee
    # solve discontinuities
    axis_list = np.array(axis_list)
    angle_list = np.array(angle_list)
    [axis, angle] = solve_discontinuities(axis_list, angle_list, flg_plot=False)
    axis_angle = matlib.repmat(angle,3,1).T*axis
    trajectory[:,3:6] = axis_angle

    return trajectory

label = 'tight_dual_pick_box'
# label = 'alt_turn_valve'
rate = 200.
n_basis = 200


demo = np.loadtxt('DATA/'+label+'.csv', delimiter=',')

demo = demo[0:2567,:]

time_steps = demo[:,0] # 1st columun: time
time_steps = time_steps-time_steps[0] 
print("Demo frequency: ", 1/(np.mean(time_steps[1:]-time_steps[:-1])))
x_demo = demo[:,1:] # 2nd-4th columns: VF position / 5th-8th columns: VF axis-angle / 9-th column: EE distance

dt = 1.0/rate

tau = time_steps[-1]
x0 = x_demo[0,:]
g0 = x_demo[-1,:]

print("Loading fixed rotation matrices")
R_right = np.loadtxt('DATA/'+label+'_R_right.csv', delimiter=',')
R_left = np.loadtxt('DATA/'+label+'_R_left.csv', delimiter=',')

print("Getting target frame for task: "+label)
world_to_target = np.loadtxt('DATA/'+label+'_tf.csv', delimiter=',')
world_to_target_translation = world_to_target[0:3]
world_to_target_quaternion = world_to_target[3:7]

R_world_to_target = quat_to_rot(world_to_target_quaternion)

H_world_to_target = np.zeros((4,4))
H_world_to_target[0:3,0:3] = R_world_to_target
H_world_to_target[0:3,3] = world_to_target_translation
H_world_to_target[3,3] = 1.0

tr_dmp = Bimanual_Target_DMP(num_basis=n_basis, tau=tau, dt = dt, x0=x0, g0=g0, w_H_t = H_world_to_target, R_right = R_right, R_left = R_left)
tr_x_demo, _, _, tr_demo_time = tr_dmp.learn_from_demo(x_demo, time_steps, 'savgol')
tr_time_steps, tr_p_r, tr_q_r, tr_p_l, tr_q_l, _, _, _, _, _ = tr_dmp.rollout()

tr_trajectory_r = np.zeros((tr_p_r.shape[0], 7))
tr_trajectory_r[:,:3] = tr_p_r
tr_trajectory_r[:,3:] = tr_q_r
tr_trajectory_r = change_frame(tr_trajectory_r, H_world_to_target)
tr_trajectory_l = np.zeros((tr_p_l.shape[0], 7))
tr_trajectory_l[:,:3] = tr_p_l
tr_trajectory_l[:,3:] = tr_q_l
tr_trajectory_l = change_frame(tr_trajectory_l, H_world_to_target)

wr_dmp = Bimanual_DMP(num_basis=n_basis, tau=tau, dt = dt, x0=x0, g0=g0, R_right = R_right, R_left = R_left)
wr_x_demo, _, _, wr_demo_time = wr_dmp.learn_from_demo(x_demo, time_steps, 'savgol')
wr_time_steps, wr_p_r, wr_q_r, wr_p_l, wr_q_l, _, _, _, _, _ = wr_dmp.rollout()

wr_trajectory_r = np.zeros((wr_p_r.shape[0], 7))
wr_trajectory_r[:,:3] = wr_p_r
wr_trajectory_r[:,3:] = wr_q_r
wr_trajectory_r = change_frame(wr_trajectory_r, H_world_to_target)
wr_trajectory_l = np.zeros((wr_p_l.shape[0], 7))
wr_trajectory_l[:,:3] = wr_p_l
wr_trajectory_l[:,3:] = wr_q_l
wr_trajectory_l = change_frame(wr_trajectory_l, H_world_to_target)

demo_p_r = tr_p_r
demo_p_l = tr_p_l
demo_q_r = tr_q_r
demo_q_l = tr_q_l
demo_time = tr_time_steps

demo_r = np.zeros((demo_p_r.shape[0], 7))
demo_r[:,:3] = demo_p_r
demo_r[:,3:] = demo_q_r
demo_r = change_frame(demo_r, H_world_to_target)
demo_l = np.zeros((demo_p_l.shape[0], 7))
demo_l[:,:3] = demo_p_l
demo_l[:,3:] = demo_q_l
demo_l = change_frame(demo_l, H_world_to_target)


print("Getting grapsing offset")
dmp_goal = wr_dmp.get_goal()
dmp_goal_translation = dmp_goal[0:3]
dmp_goal_axis_angle = dmp_goal[3:6]

R_world_to_target = quat_to_rot(world_to_target_quaternion)
R_world_to_dmp_goal = axis_angle_to_rot(dmp_goal_axis_angle)

H_world_to_dmp_goal = np.zeros((4,4))
H_world_to_dmp_goal[0:3,0:3] = R_world_to_dmp_goal
H_world_to_dmp_goal[0:3,3] = dmp_goal_translation
H_world_to_dmp_goal[3,3] = 1.0

H_target_to_dmp_goal = np.matmul(np.linalg.inv(H_world_to_target), H_world_to_dmp_goal)

print("Change goal")
new_world_to_target = np.loadtxt('DATA/'+'trdmp_tf.csv', delimiter=',')
new_world_to_target_translation = new_world_to_target[0:3]
new_world_to_target_quaternion = new_world_to_target[3:7]

R_new_world_to_target = quat_to_rot(new_world_to_target_quaternion)

H_new_world_to_target = np.zeros((4,4))
H_new_world_to_target[0:3,0:3] = R_new_world_to_target
H_new_world_to_target[0:3,3] = new_world_to_target_translation
H_new_world_to_target[3,3] = 1.0

tr_dmp.set_new_target_frame(H_new_world_to_target)
new_x0 = np.zeros(9)
new_x0[0:3] = (tr_p_r[0,:]+tr_p_l[0,:])/2
axis, angle = quat_to_axis_angle(tr_q_r[0,:], auto_align=True)
new_x0[3:6] = axis*angle
new_x0[6] = np.linalg.norm(tr_p_r[0,:]-tr_p_l[0,:])
tr_dmp.reset(new_x0, in_world_frame=True)


tr_time_steps, tr_p_r, tr_q_r, tr_p_l, tr_q_l, _, _, _, _, _ = tr_dmp.rollout()


H_new_goal = np.matmul(H_new_world_to_target,H_target_to_dmp_goal)
goal_position = H_new_goal[0:3,3]
goal_axis, goal_angle = rot_to_axis_angle(H_new_goal[0:3,0:3], auto_align=True)
goal_pose = np.concatenate((goal_position,goal_axis*goal_angle))
dmp_goal = wr_dmp.get_goal()
dmp_goal[0:6] = goal_pose         
wr_dmp.set_new_goal(dmp_goal)

wr_time_steps, wr_p_r, wr_q_r, wr_p_l, wr_q_l, _, _, _, _, _ = wr_dmp.rollout()


tr_trajectory_r = np.zeros((tr_p_r.shape[0], 7))
tr_trajectory_r[:,:3] = tr_p_r
tr_trajectory_r[:,3:] = tr_q_r
tr_trajectory_r = change_frame(tr_trajectory_r, H_new_world_to_target)
tr_trajectory_l = np.zeros((tr_p_l.shape[0], 7))
tr_trajectory_l[:,:3] = tr_p_l
tr_trajectory_l[:,3:] = tr_q_l
tr_trajectory_l = change_frame(tr_trajectory_l, H_new_world_to_target)

wr_trajectory_r = np.zeros((wr_p_r.shape[0], 7))
wr_trajectory_r[:,:3] = wr_p_r
wr_trajectory_r[:,3:] = wr_q_r
wr_trajectory_r = change_frame(wr_trajectory_r, H_new_world_to_target)
wr_trajectory_l = np.zeros((wr_p_l.shape[0], 7))
wr_trajectory_l[:,:3] = wr_p_l
wr_trajectory_l[:,3:] = wr_q_l
wr_trajectory_l = change_frame(wr_trajectory_l, H_new_world_to_target)


fig = plt.figure()
ax = plt.axes(projection='3d')

# plot frames
for k in range(0, tr_trajectory_r.shape[0], int(tr_trajectory_r.shape[0]/5)):
    R_right = axis_angle_to_rot(tr_trajectory_r[k,3:6])
    ax.quiver(tr_trajectory_r[k,0], tr_trajectory_r[k,1], tr_trajectory_r[k,2], R_right[0,0], R_right[1,0], R_right[2,0], colors='r', length=0.05, normalize=True)
    ax.quiver(tr_trajectory_r[k,0], tr_trajectory_r[k,1], tr_trajectory_r[k,2], R_right[0,1], R_right[1,1], R_right[2,1], colors='g', length=0.05, normalize=True)
    ax.quiver(tr_trajectory_r[k,0], tr_trajectory_r[k,1], tr_trajectory_r[k,2], R_right[0,2], R_right[1,2], R_right[2,2], colors='b', length=0.05, normalize=True)
    R_left = axis_angle_to_rot(tr_trajectory_l[k,3:6])
    ax.quiver(tr_trajectory_l[k,0], tr_trajectory_l[k,1], tr_trajectory_l[k,2], R_left[0,0], R_left[1,0], R_left[2,0], colors='r', length=0.05, normalize=True)
    ax.quiver(tr_trajectory_l[k,0], tr_trajectory_l[k,1], tr_trajectory_l[k,2], R_left[0,1], R_left[1,1], R_left[2,1], colors='g', length=0.05, normalize=True)
    ax.quiver(tr_trajectory_l[k,0], tr_trajectory_l[k,1], tr_trajectory_l[k,2], R_left[0,2], R_left[1,2], R_left[2,2], colors='b', length=0.05, normalize=True)

ax.plot3D(demo_r[:,0], demo_r[:,1], demo_r[:,2], 'tab:orange', linewidth=3,  label='Demo')
ax.plot3D(demo_l[:,0], demo_l[:,1], demo_l[:,2], 'tab:orange', linewidth=3)

ax.plot3D(wr_trajectory_r[:,0], wr_trajectory_r[:,1], wr_trajectory_r[:,2], 'tab:blue', linewidth=3,  label='WR-DMP')
ax.plot3D(wr_trajectory_l[:,0], wr_trajectory_l[:,1], wr_trajectory_l[:,2], 'tab:blue', linewidth=3)

ax.plot3D(tr_trajectory_r[:,0], tr_trajectory_r[:,1], tr_trajectory_r[:,2], 'tab:green', linewidth=3,  label='TR-DMP')
ax.plot3D(tr_trajectory_l[:,0], tr_trajectory_l[:,1], tr_trajectory_l[:,2], 'tab:green', linewidth=3)

for k in range(0, wr_trajectory_r.shape[0], int(wr_trajectory_r.shape[0]/5)):
    R_right = axis_angle_to_rot(wr_trajectory_r[k,3:6])
    ax.quiver(wr_trajectory_r[k,0], wr_trajectory_r[k,1], wr_trajectory_r[k,2], R_right[0,0], R_right[1,0], R_right[2,0], colors='r', length=0.05, normalize=True)
    ax.quiver(wr_trajectory_r[k,0], wr_trajectory_r[k,1], wr_trajectory_r[k,2], R_right[0,1], R_right[1,1], R_right[2,1], colors='g', length=0.05, normalize=True)
    ax.quiver(wr_trajectory_r[k,0], wr_trajectory_r[k,1], wr_trajectory_r[k,2], R_right[0,2], R_right[1,2], R_right[2,2], colors='b', length=0.05, normalize=True)
    R_left = axis_angle_to_rot(wr_trajectory_l[k,3:6])
    ax.quiver(wr_trajectory_l[k,0], wr_trajectory_l[k,1], wr_trajectory_l[k,2], R_left[0,0], R_left[1,0], R_left[2,0], colors='r', length=0.05, normalize=True)
    ax.quiver(wr_trajectory_l[k,0], wr_trajectory_l[k,1], wr_trajectory_l[k,2], R_left[0,1], R_left[1,1], R_left[2,1], colors='g', length=0.05, normalize=True)
    ax.quiver(wr_trajectory_l[k,0], wr_trajectory_l[k,1], wr_trajectory_l[k,2], R_left[0,2], R_left[1,2], R_left[2,2], colors='b', length=0.05, normalize=True)


for k in range(0, demo_r.shape[0], int(demo_r.shape[0]/5)):
    R_right = axis_angle_to_rot(demo_r[k,3:6])
    ax.quiver(demo_r[k,0], demo_r[k,1], demo_r[k,2], R_right[0,0], R_right[1,0], R_right[2,0], colors='r', length=0.05, normalize=True)
    ax.quiver(demo_r[k,0], demo_r[k,1], demo_r[k,2], R_right[0,1], R_right[1,1], R_right[2,1], colors='g', length=0.05, normalize=True)
    ax.quiver(demo_r[k,0], demo_r[k,1], demo_r[k,2], R_right[0,2], R_right[1,2], R_right[2,2], colors='b', length=0.05, normalize=True)
    R_left = axis_angle_to_rot(demo_l[k,3:6])
    ax.quiver(demo_l[k,0], demo_l[k,1], demo_l[k,2], R_left[0,0], R_left[1,0], R_left[2,0], colors='r', length=0.05, normalize=True)
    ax.quiver(demo_l[k,0], demo_l[k,1], demo_l[k,2], R_left[0,1], R_left[1,1], R_left[2,1], colors='g', length=0.05, normalize=True)
    ax.quiver(demo_l[k,0], demo_l[k,1], demo_l[k,2], R_left[0,2], R_left[1,2], R_left[2,2], colors='b', length=0.05, normalize=True)


R_box = axis_angle_to_rot(demo_r[k,3:6])
ax.quiver(0,0,0, 1,0,0, colors='r', length=0.05, normalize=True)
ax.quiver(0,0,0, 0,1,0, colors='g', length=0.05, normalize=True)
ax.quiver(0,0,0, 0,0,1, colors='b', length=0.05, normalize=True)

ax.scatter3D(tr_trajectory_r[0,0], tr_trajectory_r[0,1], tr_trajectory_r[0,2], color='tab:green', linewidth=3)
ax.scatter3D(tr_trajectory_l[0,0], tr_trajectory_l[0,1], tr_trajectory_l[0,2], color='tab:green', linewidth=3)
ax.scatter3D(wr_trajectory_r[0,0], wr_trajectory_r[0,1], wr_trajectory_r[0,2], color='tab:blue', linewidth=3)
ax.scatter3D(wr_trajectory_l[0,0], wr_trajectory_l[0,1], wr_trajectory_l[0,2], color='tab:blue', linewidth=3)
ax.scatter3D(demo_r[0,0], demo_r[0,1], demo_r[0,2], color='tab:orange', linewidth=3)
ax.scatter3D(demo_l[0,0], demo_l[0,1], demo_l[0,2], color='tab:orange', linewidth=3)



ax.set_aspect('equal', adjustable='box')
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
ax.elev = 0
ax.azim = 0  
plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=3)

plt.grid()
plt.show()