"""Geometry module

This module contains a set of util functions for handling different 
rotation representations:

    * solve_discontinuities - solve discontinuities in axis-angle traj.
    * rot_to_quat - rotation matrix to quaternion conversion
    * quat_to_rot - quaternion to rotation matrix conversion
    * rot_to_axis_angle - rotation matrix to axis-angle conversion
    * axis_angle_to_rot - axis-angle to rotation matrix conversion
    * axis_angle_to_quat - axis-angle to quaternion conversion
    * quat_to_axis_angle - quaternion to axis-angle conversion
    * skew - get skew matrix form of a 3-dimensional vector
    * quat_product - quaternion product
    * quat_conj - quaternion conjugation
"""

import numpy as np
import matplotlib.pyplot as plt


def solve_discontinuities(axis, angle, flg_plot = False):
    """Solve discontinuities in the given axis angle trajectory.

    Parameters
    ----------
    axis : numpy.ndarray
        trajectory of axis values [axis.shape : (num_step, 3)]
    angle : numpy.ndarray
        trajectory of angle values [angle.shape : (num_step,)]
    flg_plot : bool, optional
        flag to show the trajectory plot (default is False)

    Returns
    ------
    list
        a list of numpy.ndarray with the continuous axis and angle

    Raises
    ------
    ValueError
        If axis and angle trajectories have different lenght
        If axis 2nd dimension is not 3
    """

    num_steps = len(angle)
    if axis.shape[0] != num_steps:
        raise ValueError("axis and angle must have the same lenght")
    if axis.shape[1] != 3:
        raise ValueError("axis 2nd dimension must be 3")

    # plot original trajectory
    if flg_plot:
        fig, axs = plt.subplots(nrows = 4, ncols = 1)
        for j in range(3):
            axs[j].plot(axis[:,j])
        axs[-1].plot(angle)

    # solve discontinuities
    for k in range(1, num_steps):
        if ( np.linalg.norm(axis[k-1,:] - axis[k,:]) > np.linalg.norm( \
            axis[k-1,:] + axis[k,:]) ):

            axis[k,:] = -axis[k,:]
            angle[k] = -angle[k] + np.sign(angle[k]) * 2 * np.pi
        
    # plot continuous trajectory
    if flg_plot:
        for j in range(3):
            axs[j].plot(axis[:,j])
            axs[j].grid()
        axs[-1].plot(angle)
        axs[-1].grid()
        axs[0].set_title('Solve discontinuities')
        axs[0].set(ylabel=r'$x$')
        axs[1].set(ylabel=r'$y$')
        axs[2].set(ylabel=r'$z$')
        axs[3].set(ylabel=r'$\theta$')
        fig.tight_layout()
        plt.show()

    return axis, angle

def is_rotation_matrix(R):
    """Check if a matrix is a valid rotation matrix.

    Parameters
    ----------
    R : numpy.ndarray
        input matrix [R.shape : (3, 3)]

    Returns
    ------
    bool
        input matrix is a valid rotation matrix

    Raises
    ------
    ValueError
        If R is not a 3-by-3 matrix
    """
    R = np.array(R)
    if R.shape != (3,3):
        raise ValueError("R must be a 3-by-3 matrix")

    epsilon = 10**-6

    I = np.eye(3,3)
    if ( np.abs(np.linalg.det(R) - 1) > epsilon or 
        np.sum(I - np.matmul(R, R.T)) > epsilon ):
        return False
    else
        return True

def rot_to_quat(R):
    """Rotation matrix to quaternion conversion.

    Parameters
    ----------
    R : numpy.ndarray
        rotation matrix [R.shape : (3, 3)]

    Returns
    ------
    list
        a list containing qx, qy, qz, qw [float]

    Raises
    ------
    ValueError
        If R is not a valid rotation matrix
    """
    
    R = np.array(R)
    if not is_rotation_matrix(R):
        raise ValueError("R must be a valid rotation matrix")

    epsilon = 10**-6

    tr = np.trace(R)

    if tr > epsilon:

        sqrt_tr = np.sqrt(tr + 1.)
        qw = 0.5 * sqrt_tr
        qx = (R[2,1] - R[1,2]) / (2. * sqrt_tr)
        qy = (R[0,2] - R[2,0]) / (2. * sqrt_tr)
        qz = (R[1,0] - R[0,1]) / (2. * sqrt_tr)

    elif (R[1,1] > R[0,0]) and (R[1,1] > R[2,2]):

        # max value at R[1,1]
        sqrt_tr = np.sqrt(R[1,1] - R[0,0] - R[2,2] + 1.)

        qy = 0.5*sqrt_tr

        if sqrt_tr > epsilon:
            sqrt_tr = 0.5/sqrt_tr

        qw = (R[0,2] - R[2,0]) * sqrt_tr
        qx = (R[1,0] + R[0,1]) * sqrt_tr
        qz = (R[2,1] + R[1,2]) * sqrt_tr

    elif R[2,2] > R[0,0]:

        # max value at R[2,2]
        sqrt_tr = np.sqrt(R[2,2] - R[0,0] - R[1,1] + 1.)

        qz = 0.5*sqrt_tr

        if sqrt_tr > epsilon:
            sqrt_tr = 0.5/sqrt_tr

        qw = (R[1,0] - R[0,1]) * sqrt_tr
        qx = (R[0,2] + R[2,0]) * sqrt_tr
        qy = (R[2,1] + R[1,2]) * sqrt_tr

    else:

        # max value at R[0,0]
        sqrt_tr = np.sqrt(R[0,0] - R[1,1] - R[2,2] + 1.)

        qx = 0.5 * sqrt_tr

        if sqrt_tr > epsilon:
            sqrt_tr = 0.5 / sqrt_tr

        qw = (R[2,1] - R[1,2]) * sqrt_tr
        qy = (R[1,0] + R[0,1]) * sqrt_tr
        qz = (R[0,2] + R[2,0]) * sqrt_tr

    return qx, qy, qz, qw


def quat_to_rot(quat):
    """Quaternion to rotation matrix conversion.

    Parameters
    ----------
    quat : list
        a list containing qx, qy, qz, qw [float]

    Returns
    ------
    R : numpy.ndarray
        rotation matrix [R.shape : (3, 3)]

    Raises
    ------
    ValueError
        If input list dimension is not 4
    """
    if len(quat) != 4:
        raise ValueError("Wrong input size")

    # check normalization
    quat = np.array(quat)
    norm = np.linalg.norm(quat)
    if  (1 - norm)**2 > 10**-4: # quat is not normalized
        quat = list(quat / np.linalg.norm(quat))

    qx, qy, qz, qw = quat

    R = np.zeros((3,3))
    R[0,0] = 1 - 2 * qy**2 - 2 * qz**2
    R[0,1] = 2 * qx * qy - 2 * qz * qw
    R[0,2] = 2 * qx * qz + 2 * qy * qw
    R[1,0] = 2 * qx * qy + 2 * qz * qw
    R[1,1] = 1 - 2 * qx**2 - 2 * qz**2
    R[1,2] = 2 * qy * qz - 2 * qx * qw
    R[2,0] = 2 * qx * qz - 2 * qy * qw
    R[2,1] = 2 * qy * qz + 2 * qx * qw
    R[2,2] = 1 - 2 * qx**2 - 2 * qy**2

    return R


def rot_to_axis_angle(R, auto_align = False, ref_axis = np.array([1,0,0])):
    """Rotation matrix to axis-angle conversion.

    Parameters
    ----------
    R : numpy.ndarray
        rotation matrix [R.shape : (3, 3)]
    auto_align : bool, optional
        flag to align axis-angle with reference axis (default is False)
    ref_axis : numpy.ndarray, optional
        reference axis for auto alignment (default is np.array([1,0,0]))

    Returns
    ------
    list
        a list containing axis [numpy.ndarray] and angle [float]

    Raises
    ------
    ValueError
        If R is not a valid rotation matrix
    """
    R = np.array(R)
    if not is_rotation_matrix(R):
        raise ValueError("R must be a valid rotation matrix")

    axis, angle = quat_to_axis_angle(rot_to_quat(R), auto_align = auto_align, 
                                                     ref_axis = ref_axis)

    return axis, angle


# def rot_to_axis_angle(R, auto_align = False, ref_axis = np.array([1,0,0])):
#     """Rotation matrix to axis-angle conversion.

#     Parameters
#     ----------
#     R : numpy.ndarray
#         rotation matrix [R.shape : (3, 3)]
#     auto_align : bool, optional
#         flag to align axis-angle with reference axis (default is False)
#     ref_axis : numpy.ndarray, optional
#         reference axis for auto alignment (default is np.array([1,0,0]))

#     Returns
#     ------
#     list
#         a list containing axis [numpy.ndarray] and angle [float]

#     Raises
#     ------
#     ValueError
#         If R is not a valid rotation matrix
#     """
#     R = np.array(R)
#     if not is_rotation_matrix(R):
#         raise ValueError("R must be a valid rotation matrix")

#     epsilon_round = 0.01 # rounding error margin
#     epsilon_singularity = 0.1 # margin to distinguish between 0 and 180 deg

#     if (abs(R[0,1] - R[1,0]) < epsilon_round and
#         abs(R[0,2] - R[2,0]) < epsilon_round and
#         abs(R[1,2] - R[1,0]) < epsilon_round):
#         # singularity found

#         if (abs(R[0,1] - R[1,0]) < epsilon_singularity and
#             abs(R[0,2] - R[2,0]) < epsilon_singularity and
#             abs(R[1,2] - R[1,0]) < epsilon_singularity and
#             abs(R[0,0] + R[1,1] + R[2,2]-3) < epsilon_singularity):

#             angle = 0
#             axis = np.array([1,0,0])

#         else:

#             angle = np.pi
#             xx = (R[0,0] + 1) / 2
#             yy = (R[1,1] + 1) / 2
#             zz = (R[2,2] + 1) / 2
#             xy = (R[0,1] + R[1,0]) / 4
#             xz = (R[0,2] + R[2,0]) / 4
#             yz = (R[1,2] + R[2,1]) / 4

#             if (xx > yy and xx > zz):
#                 # R[0][0] is the largest diagonal term
#                 if (xx < epsilon_round):
#                     x = 0
#                     y = 1 / np.sqrt(2)
#                     z = 1 / np.sqrt(2)
#                 else:
#                     x = np.sqrt(xx)
#                     y = xy / x
#                     z = xz / x
#             elif (yy > zz):
#                 # R[1][1] is the largest diagonal term
#                 if (yy < epsilon_round):
#                     x = 1 / np.sqrt(2)
#                     y = 0
#                     z = 1 / np.sqrt(2)
#                 else:
#                     y = np.sqrt(yy)
#                     x = xy / y
#                     z = yz / y
#             else:
#                 # R[2][2] is the largest diagonal term
#                 if (zz < epsilon_round):
#                     x = 1 / np.sqrt(2)
#                     y = 1 / np.sqrt(2)
#                     z = 0
#                 else:
#                     z = np.sqrt(zz)
#                     x = xz / z
#                     y = yz / z

#             axis = np.array([x,y,z])

#     else: # no singularities
#         s = np.sqrt((R[2,1] - R[1,2]) * (R[2,1] - R[1,2]) \
#             + (R[0,2] - R[2,0]) * (R[0,2] - R[2,0]) \
#             + (R[1,0] - R[0,1]) * (R[1,0] - R[0,1])) # used to normalise

#         angle = np.arccos((R[0,0] + R[1,1] + R[2,2] -1) / 2)
#         x = (R[2,1] - R[1,2]) / s
#         y = (R[0,2] - R[2,0]) / s
#         z = (R[1,0] - R[0,1]) / s
#         axis = np.array([x,y,z])

#     if auto_align:
#         if np.dot(axis,ref_axis) < 0.0:
#             axis = - axis
#             angle = - angle + 2 * np.pi

#     return axis, angle


def axis_angle_to_rot(axis_angle):
    """Axis-angle to rotation matrix conversion.

    Parameters
    ----------
    axis_angle : numpy.ndarray
        rotation matrix [axis_angle.shape : (3,)]

    Returns
    ------
    numpy.ndarray
        rotation matrix conversion

    Raises
    ------
    ValueError
        If input list dimension is not 3
    """
    if len(axis_angle) != 3:
        raise ValueError("Wrong input size")

    angle = np.linalg.norm(axis_angle)
    if angle < 10**-4:
        R = np.eye(3)
        return R
    else:
        axis = axis_angle / angle

        c = np.cos(angle)
        s = np.sin(angle)
        t = 1 - c
        x = axis[0]
        y = axis[1]
        z = axis[2]

        R = np.zeros((3,3))
        R[0,0] = t * x**2 + c
        R[0,1] = t * x * y - z * s
        R[0,2] = t * x * z + y * s
        R[1,0] = t * x * y + z * s
        R[1,1] = t * y **2 + c
        R[1,2] = t * y * z - x * s
        R[2,0] = t * x * z - y * s
        R[2,1] = t * y * z + x * s
        R[2,2] = t * z**2 + c

        return R


def axis_angle_to_quat(axis_angle):
    """Axis-angle to quaternion conversion.

    Parameters
    ----------
    axis_angle : numpy.ndarray
        rotation matrix [axis_angle.shape : (3,)]

    Returns
    ------
    list
        a list containing qx, qy, qz, qw [float]

    Raises
    ------
    list
        If input list dimension is not 3
    """
    if len(axis_angle) != 3:
        raise ValueError("Wrong input size")

    angle = np.linalg.norm(axis_angle)
    
    if angle < 10**-4: # angle = 0
        qx = 0.
        qy = 0.
        qz = 0.
        qw = 1.

    else:

        axis = axis_angle / angle

        qx = axis[0] * np.sin(angle / 2)
        qy = axis[1] * np.sin(angle / 2)
        qz = axis[2] * np.sin(angle / 2)
        qw = np.cos(angle / 2)

    return qx, qy, qz, qw


def quat_to_axis_angle(quat, auto_align = False, ref_axis = np.array([1,0,0])):
    """Quaternion to rotation matrix conversion.

    Parameters
    ----------
    quat : list
        a list containing qx, qy, qz, qw [float]
    auto_align : bool, optional
        flag to align axis-angle with reference axis (default is False)
    ref_axis : numpy.ndarray, optional
        reference axis for auto alignment (default is np.array([1,0,0]))

    Returns
    ------
    list
        a list containing axis [numpy.ndarray] and angle [float]

    Raises
    ------
    ValueError
        If input list dimension is not 4
    """
    if len(quat) != 4:
        raise ValueError("Wrong input size")

    # check normalization
    quat = np.array(quat)
    norm = np.linalg.norm(quat)
    if  (1 - norm)**2 > 10**-4: # quat is not normalized
        quat = list(quat / np.linalg.norm(quat))

    qx, qy, qz, qw = quat

    angle = 2 * np.arccos(qw)

    s = np.sqrt(1 - qw * qw)

    if s < 10**-4: # angle = 0
        x = 1.
        y = 0.
        z = 0.
    else:
        x = qx / np.sqrt(1 - qw * qw)
        y = qy / np.sqrt(1 - qw * qw)
        z = qz / np.sqrt(1 - qw * qw)

    axis = np.array([x,y,z])

    if auto_align:
        if np.dot(axis,ref_axis) < 0.0:
            axis = - axis
            angle = - angle + 2 * np.pi

    return axis, angle


def skew(v):
    """Compute skew matrix form of 3D vector.

    Parameters
    ----------
    v : numpy.ndarray
        vector [v.shape = (3,)]

    Returns
    ------
    numpy.ndarray
        skew matrix form

    Raises
    ------
    ValueError
        If v dimension is not 3
    """
    if len(v) != 3:
        raise ValueError("Wrong input size")

    s = np.zeros((3,3))
    s[0,1] = -v[2]
    s[0,2] =  v[1]
    s[1,0] =  v[2]
    s[1,2] = -v[0]
    s[2,0] = -v[1]
    s[2,1] =  v[0]
    return s



def quat_product(q1, q2):
    """Quaternion product.

    Parameters
    ----------
    q1 : list
        a list containing qx, qy, qz, qw [float] of 1st quaternion
    q2 : list
        a list containing qx, qy, qz, qw [float] of 2nd quaternion

    Returns
    ------
    list
        a list containing qx, qy, qz, qw [float] of product quaternion

    Raises
    ------
    ValueError
        If q1 dimension is not 4
        If q2 dimension is not 4
    """
    if len(q1) != 4:
        raise ValueError("Wrong input size")
    if len(q2) != 4:
        raise ValueError("Wrong input size")

    q1 = np.array(q1)
    q2 = np.array(q2)

    # check normalization
    norm_1 = np.linalg.norm(q1)
    norm_2 = np.linalg.norm(q2)
    if  (1 - norm_1)**2 > 10**-4: # q1 is not normalized
        q1 = q1 / norm_1

    if (1 - norm_2)**2 > 10**-4: # q2 is not normalized
        q2 = q2 / norm_2      
 
    q1w = q1[-1]
    q2w = q2[-1]
    v1 = q1[0:3].T
    v2 = q2[0:3].T
    prod_w = q1w * q2w - np.matmul(v1.T, v2)
    prod_v = q1w * v2 + q2w * v1 + np.cross(v1, v2)

    return prod_v[0], prod_v[1], prod_v[2], prod_w


def quat_conj(quat):
    """Quaternion conjugation.

    Parameters
    ----------
    quat : list
        a list containing qx, qy, qz, qw [float]

    Returns
    ------
    list
        a list containing qx, qy, qz, qw [float] of conjugate quaternion

    Raises
    ------
    ValueError
        If input list dimension is not 4
    """
    if len(quat) != 4:
        raise ValueError("Wrong input size")

    quat = np.array(quat)

    # check normalization
    quat = np.array(quat)
    norm = np.linalg.norm(quat)
    if  (1 - norm)**2 > 10**-4: # quat is not normalized
        quat = list(quat / np.linalg.norm(quat))

    return -quat[0], -quat[1], -quat[2], quat[3]