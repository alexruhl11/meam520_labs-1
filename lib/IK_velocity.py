import numpy as np
from lib.calcJacobian import calcJacobian


def IK_velocity(q_in, v_in, omega_in):
    """
    :param q: 0 x 7 vector corresponding to the robot's current configuration.
    :param v: The desired linear velocity in the world frame. If any element is
    Nan, then that velocity can be anything
    :param omega: The desired angular velocity in the world frame. If any
    element is Nan, then that velocity is unconstrained i.e. it can be anything
    :return:
    dq - 0 x 7 vector corresponding to the joint velocities. If v and omega
         are infeasible, then dq should minimize the least squares error. If v
         and omega have multiple solutions, then you should select the solution
         that minimizes the l2 norm of dq
    """

    ## STUDENT CODE GOES HERE
    j = calcJacobian(q_in)
    b = np.concatenate((v_in, omega_in)).reshape((-1, 1))
    aug = np.concatenate((j, b), axis=1)
    aug = aug[~np.isnan(aug).any(axis=1)]
    j = aug[:, :-1]
    b = aug[:, -1]
    dq, _, _, _ = np.linalg.lstsq(j, b)
    return dq
