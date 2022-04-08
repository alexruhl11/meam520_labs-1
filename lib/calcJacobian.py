import numpy as np
from lib.calculateFK import FK

def calcJacobian(q):
    """
    Calculate the full Jacobian of the end effector in a given configuration
    :param q: 0 x 7 configuration vector (of joint angles) [q0,q1,q2,q3,q4,q5,q6]
    :return: J - 6 x 7 matrix representing the Jacobian, where the first three
    rows correspond to the linear velocity and the last three rows correspond to
    the angular velocity, expressed in world frame coordinates
    """
    fk = FK()
    J = np.zeros((6, 7))

    ai_list = fk.compute_Ai(q) #list of ai for all q

    ai = np.identity(4)
    tranform_matrix = [ai]
    for i in range(len(ai_list)):
        ai = ai @ ai_list[i]
        tranform_matrix.append(ai)

    o_n = []
    z_n = []
    for i in range(8):
        o_n.append(tranform_matrix[i][:3, 3])
        z_n.append(tranform_matrix[i][:3, 2])

    for i in range(7):
        J[:3, i] = np.cross(z_n[i], o_n[7] - o_n[i])
        J[3:, i] = z_n[i]

    return J

if __name__ == '__main__':
    q= np.array([0, 0, 0, -np.pi/2, 0, np.pi/2, np.pi/4])
    print(np.round(calcJacobian(q),3))
