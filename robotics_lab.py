"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

ROBOTICS LAB
Free gift to the scientific community from QuLabInfinite.
"""

from dataclasses import dataclass, field
import numpy as np
from scipy.constants import k, Avogadro, g, c, h, e, pi

@dataclass
class Robot:
    """
    Main robotics class for simulation and control of robotic systems.
    """
    mass: float = 1.0
    length_arm: float = 0.5
    num_joints: int = 3
    gravity_acceleration: float = g
    
    def __post_init__(self):
        self.joint_angles = np.zeros((self.num_joints, ), dtype=np.float64)
        self.end_effector_position = None

    def forward_kinematics(self) -> np.ndarray:
        """
        Computes the forward kinematics of a robotic arm.
        Returns the position of the end effector in 3D space.
        """
        if not self.num_joints > 0 or self.length_arm <= 0.0:
            return np.array([np.nan, np.nan, np.nan], dtype=np.float64)
        
        pos = np.zeros((1, 3), dtype=np.float64) 
        for joint in range(self.num_joints):
            offset = self.joint_angles[joint]
            pos[0] += [-self.length_arm * np.sin(offset),
                        -self.length_arm * np.cos(offset),
                        -self.length_arm / 2.0]
            
        return pos[0]

    def inverse_kinematics(self, target_position: np.ndarray) -> np.ndarray:
        """
        Computes the inverse kinematics for a robotic arm.
        Takes in the desired end effector position and returns the joint angles required to achieve it.
        """
        if not self.num_joints > 0 or self.length_arm <= 0.0:
            return np.array([np.nan, ] * self.num_joints)
        
        delta = target_position - np.zeros((1,3), dtype=np.float64)[0]
        for i in range(self.num_joints):
            angle = np.arctan2(delta[0], delta[1])
            self.joint_angles[i] = angle
        return self.joint_angles

    def calculate_dynamics(self) -> np.ndarray:
        """
        Computes the dynamics of the robotic arm, including torques at joints.
        Returns an array with joint torques [tau_1, tau_2, ..., tau_n].
        """
        if not self.num_joints > 0 or self.length_arm <= 0.0:
            return np.array([np.nan, ] * self.num_joints)
        
        torque = np.zeros((self.num_joints,), dtype=np.float64) 
        for joint in range(self.num_joints):
            torque[joint] = -self.mass * g
        return torque

def run_demo():
    # Initialize robot with parameters.
    my_robot = Robot(mass=1.0, length_arm=0.5, num_joints=3)

    print("Initial robot state:")
    print(f"Joint Angles: {my_robot.joint_angles}")
    
    # Set target position for inverse kinematics.
    target_position = np.array([1.5, 2.5, -0.5], dtype=np.float64)
    
    joint_angles = my_robot.inverse_kinematics(target_position)

    print("\nTarget Position:")
    print(f"Joint Angles: {joint_angles}")
    
    # Calculate new end-effector position using forward kinematics.
    pos = my_robot.forward_kinematics()
    
    print("\nNew End-Effector Position (Forward Kinematics):")
    print(f"Position: {pos}")

if __name__ == '__main__':
    run_demo()