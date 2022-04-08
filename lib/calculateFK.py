import numpy as np

np.set_printoptions(suppress=True)
from math import pi, cos, sin


class FK():

    def __init__(self):

        # TODO: you may want to define geometric parameters here that will be
        # useful in computing the forward kinematics. The data you will need
        # is provided in the lab handout
        self.a = [0, 0, 0.0825, -0.0825, 0, 0.088, 0, 0]
        self.d = [0.192, 0, 0.195 + 0.121, 0, 0.125 + 0.259, 0, 0.051 + 0.159, 0]
        self.alpha = [-pi / 2, pi / 2, pi / 2, -pi / 2, pi / 2, pi / 2, 0, 0]
        self.theta_offset = [0, 0, 0, 0, 0, 0, -pi / 4 + pi / 2, 0]
        self.frame_pos = np.array([
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0.195, 1],
            [0, 0, 0, 1],
            [0, 0, 0.125, 1],
            [0, 0, -0.015, 1],
            [0, 0, .051, 1],
            [0, 0, 0, 1],  # NOT AFFECTING THE END DEFFECTOR
        ])

    def forward(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        jointPositions - 8 x 3 matrix, where each row corresponds to a rotational joint of the robot or end effector
                  Each row contains the [x,y,z] coordinates in the world frame of the respective joint's center in meters.
                  The base of the robot is located at [0,0,0].
        T0e       - a 4 x 4 homogeneous transformation matrix,
                  representing the end effector frame expressed in the
                  world frame
        """

        # Your Lab 1 code starts here

        jointPositions = np.zeros((8, 3))
        T0e = np.identity(4)

        # Your code ends here
        calc_A = lambda theta, alpha, d, a: np.array(
            [
                [cos(theta), -sin(theta) * cos(alpha), sin(theta) * sin(alpha), a * cos(theta)],
                [sin(theta), cos(theta) * cos(alpha), -cos(theta) * sin(alpha), a * sin(theta)],
                [0, sin(alpha), cos(alpha), d],
                [0, 0, 0, 1]
            ]
        )

        # First Transformation Matrix from world frame to first joint
        # Note for this, theta = 0, alpha = 0, a = 0, and d = 0.141
        a_i = calc_A(0, 0, 0.141, 0)
        T0e = T0e @ a_i
        P = T0e @ self.frame_pos[0]
        jointPositions[0] = P[:-1]

        for i in range(len(q)):
            theta = q[i]
            a_i = calc_A(theta - self.theta_offset[i], self.alpha[i], self.d[i], self.a[i])
            T0e = T0e @ a_i

            P = T0e @ self.frame_pos[i+1]
            jointPositions[i+1] = P[:-1]

        return jointPositions, T0e

    # feel free to define additional helper methods to modularize your solution for lab 1

    # This code is for Lab 2, you can ignore it ofr Lab 1
    def get_axis_of_rotation(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        axis_of_rotation_list: - 3x7 np array of unit vectors describing the axis of rotation for each joint in the world frame

        """


    def compute_Ai(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        Ai: - 4x4 list of np array of homogenous transformations describing the FK of the robot. Transformations are not necessarily located at the joint locations
        """
        ai_list = []

        calc_Ai = lambda theta, alpha, d, a: np.array(
            [
                [cos(theta), -sin(theta) * cos(alpha), sin(theta) * sin(alpha), a * cos(theta)],
                [sin(theta), cos(theta) * cos(alpha), -cos(theta) * sin(alpha), a * sin(theta)],
                [0, sin(alpha), cos(alpha), d],
                [0, 0, 0, 1]
            ]
        )
        for i in range(len(q)):
            theta = q[i]
            a_i = calc_Ai(theta - self.theta_offset[i], self.alpha[i], self.d[i], self.a[i])
            ai_list.append(a_i)
        return ai_list



if __name__ == "__main__":
    fk = FK()

    # matches figure in the handout
    q = np.array([0, 0, 0, -pi / 2, 0, pi / 2, pi / 4])
    joint_positions, T0e = fk.forward(q)
    print("Joint Positions At rest:\n", joint_positions.astype(float))
    print("\n\n")
    q = np.array([0, -pi / 2, 0, -pi / 2, 0, pi / 2, pi / 4])
    joint_positions, T0e = fk.forward(q)
    print("New Joint Positions:\n", joint_positions.astype(float))
    print("End Effector Pose:\n", T0e.astype(float))

