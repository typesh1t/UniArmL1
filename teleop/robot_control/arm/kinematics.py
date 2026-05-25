# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np


class RobotKinematics:
    """Robot kinematics using placo library for forward and inverse kinematics."""

    def __init__(
        self,
        urdf_path: str,
        target_frame_name: str = "gripper_frame_link",
        joint_names: list[str] | None = None,
    ):
        """
        Initialize placo-based kinematics solver.

        Args:
            urdf_path (str): Path to the robot URDF file
            target_frame_name (str): Name of the end-effector frame in the URDF
            joint_names (list[str] | None): List of joint names to use for the kinematics solver
        """
        try:
            import placo  # type: ignore[import-not-found] # C++ library with Python bindings, no type stubs available. TODO: Create stub file or request upstream typing support.
        except ImportError as e:
            raise ImportError(
                "placo is required for RobotKinematics. "
                "Please install the optional dependencies of `kinematics` in the package."
            ) from e

        # 确保 urdf_path 是字符串（placo 不接受 PosixPath）
        urdf_path_str = str(urdf_path)
        self.robot = placo.RobotWrapper(urdf_path_str)
        self.solver = placo.KinematicsSolver(self.robot)
        self.solver.mask_fbase(True)  # Fix the base

        self.target_frame_name = target_frame_name

        # Set joint names
        self.joint_names = list(self.robot.joint_names()) if joint_names is None else joint_names

        # Initialize frame task for IK
        self.tip_frame = self.solver.add_frame_task(self.target_frame_name, np.eye(4))
        # self.tip_frame = self.solver.add_frame_task(self.target_frame_name, np.eye(4), placo.FrameType.body)

        # 关节偏好任务：让某些关节更倾向于移动
        # 格式: {关节名: 权重}，权重越大越倾向于移动该关节
        self.joint_preferences = {}

        # 平滑正则化任务（可选）
        self.smoothing_task = None
        self.smoothing_weight = 0.0

    def set_joint_preference(self, joint_name: str, weight: float):
        """
        设置关节偏好权重，让IK求解时更倾向于使用该关节

        Args:
            joint_name: 关节名称
            weight: 偏好权重（越大越倾向于移动该关节）
        """
        self.joint_preferences[joint_name] = weight

    def set_joint_preferences(self, preferences: dict[str, float]):
        """
        批量设置关节偏好权重

        Args:
            preferences: {关节名: 权重} 字典
        """
        self.joint_preferences.update(preferences)

    def set_smoothing_weight(self, weight: float):
        """
        设置平滑正则化权重

        Args:
            weight: 平滑权重（越大越平滑，但响应越慢）
                    典型值: 0.01-0.1
        """
        self.smoothing_weight = weight

    def forward_kinematics(self, joint_pos_deg: np.ndarray) -> np.ndarray:
        """
        Compute forward kinematics for given joint configuration given the target frame name in the constructor.

        Args:
            joint_pos_deg: Joint positions in degrees (numpy array)

        Returns:
            4x4 transformation matrix of the end-effector pose
        """

        # Convert degrees to radians
        joint_pos_rad = np.deg2rad(joint_pos_deg[: len(self.joint_names)])

        # Update joint positions in placo robot
        for i, joint_name in enumerate(self.joint_names):
            self.robot.set_joint(joint_name, joint_pos_rad[i])

        # Update kinematics
        self.robot.update_kinematics()

        # Get the transformation matrix
        return self.robot.get_T_world_frame(self.target_frame_name)

    def inverse_kinematics(
        self,
        current_joint_pos: np.ndarray,
        desired_ee_pose: np.ndarray,
        position_weight: float = 1.0,
        orientation_weight: float = 0.01,
        joint_regularization_weight: float = 0.01,
        smoothing_weight: float | None = None,
    ) -> np.ndarray:
        """
        Compute inverse kinematics using placo solver.

        Args:
            current_joint_pos: Current joint positions in degrees (used as initial guess)
            desired_ee_pose: Target end-effector pose as a 4x4 transformation matrix
            position_weight: Weight for position constraint in IK
            orientation_weight: Weight for orientation constraint in IK, set to 0.0 to only constrain position
            joint_regularization_weight: 正则化权重，让没有偏好的关节保持当前位置
            smoothing_weight: 平滑正则化权重（可选），越大越平滑

        Returns:
            Joint positions in degrees that achieve the desired end-effector pose
        """

        # Convert current joint positions to radians for initial guess
        current_joint_rad = np.deg2rad(current_joint_pos[: len(self.joint_names)])

        # Set current joint positions as initial guess
        for i, joint_name in enumerate(self.joint_names):
            self.robot.set_joint(joint_name, current_joint_rad[i])

        # Update the target pose for the frame task
        self.tip_frame.T_world_frame = desired_ee_pose

        # Configure the task based on position_only flag
        self.tip_frame.configure(self.target_frame_name, "soft", position_weight, orientation_weight)

        # 添加关节正则化任务：让所有关节都有正则化约束
        # 偏好权重高的关节正则化弱（容易移动），偏好权重低或无偏好的关节正则化强（难移动）
        joints_task = self.solver.add_joints_task()
        for i, joint_name in enumerate(self.joint_names):
            # 为所有关节设置正则化目标（当前位置）
            joints_task.set_joint(joint_name, current_joint_rad[i])

        # 计算每个关节的正则化权重
        # 偏好权重越高，正则化权重越低（更容易偏离当前位置）
        regularization_weights = {}
        for joint_name in self.joint_names:
            preference = self.joint_preferences.get(joint_name, 0.0)
            # 正则化权重 = 基础权重 / (偏好权重 + 1)
            # 偏好=0 → 正则化=基础权重（最强约束）
            # 偏好=0.8 → 正则化=基础权重/1.8（较弱约束）
            regularization_weights[joint_name] = joint_regularization_weight / (preference + 1.0)

        # 设置正则化权重（使用平均权重）
        avg_weight = float(np.mean(list(regularization_weights.values())))
        joints_task.configure("joints_regularization", "soft", avg_weight)

        # 添加平滑正则化任务（可选）
        smoothing_task = None
        if smoothing_weight is not None and smoothing_weight > 0:
            try:
                import placo
                smoothing_task = placo.RegularizationTask()
                smoothing_task.configure("smoothing", "soft", smoothing_weight)
                self.solver.add_task(smoothing_task)
            except Exception as e:
                print(f"Warning: Failed to add smoothing task: {e}")

        # Solve IK
        self.solver.solve(True)
        self.robot.update_kinematics()

        # 清理任务
        self.solver.remove_task(joints_task)
        if smoothing_task is not None:
            self.solver.remove_task(smoothing_task)

        # Extract joint positions
        joint_pos_rad = []
        for joint_name in self.joint_names:
            joint = self.robot.get_joint(joint_name)
            joint_pos_rad.append(joint)

        # Convert back to degrees
        joint_pos_deg = np.rad2deg(joint_pos_rad)

        # Preserve gripper position if present in current_joint_pos
        if len(current_joint_pos) > len(self.joint_names):
            result = np.zeros_like(current_joint_pos)
            result[: len(self.joint_names)] = joint_pos_deg
            result[len(self.joint_names) :] = current_joint_pos[len(self.joint_names) :]
            return result
        else:
            return joint_pos_deg
