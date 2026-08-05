"""
gripper.py

Quản lý toàn bộ logic điều khiển tay kẹp (GripperController):
- Hỗ trợ mimic joints cho Robotiq 85 và SusGrip 2F.
- Cấu hình ma sát cao (HIGH_FRICTION_LINKS) trên ngón kẹp.
- Điều khiển độ mở (move_gripper) và đóng kẹp thông minh đến khi va chạm (close_until_contact).
- Kiểm tra kẹp giữa 2 ngón (is_grasping).
"""

import math
from typing import Dict, List, Optional
import numpy as np
import pybullet as p

HIGH_FRICTION_LINKS = {
    "sus2f_pad_l_link",
    "sus2f_pad_r_link",
    "sus2f_passive_pad_l_link",
    "sus2f_passive_pad_r_link",
    "sus2f_finger_l_link",
    "sus2f_finger_r_link",
    "robotiq_85_left_finger_pad",
    "robotiq_85_right_finger_pad",
}


class GripperController:
    def __init__(
        self,
        robot_id: int,
        joints: List,
        joint_name_to_index: Dict[str, int],
        link_name_to_index: Dict[str, int],
    ):
        self.robot_id = robot_id
        self.joints = joints
        self._joint_name_to_index = joint_name_to_index
        self._link_name_to_index = link_name_to_index

        self.mimic_parent_id: Optional[int] = None
        self.mimic_child_info: Dict[int, tuple] = {}
        self.gripper_range = [0.0, 0.085]
        self.clamped_grip_val = self.gripper_range[0]

        self._setup_mimic_joints()

    def _setup_mimic_joints(self):
        """Tie child gripper joints to parent joint (Robotiq 85 & SusGrip 2F)."""
        mimic_parents = [j for j in self.joints if j.name in ("finger_joint", "gripper_joint", "base_slider_l_joint")]
        if not mimic_parents:
            self.mimic_parent_id = None
            self.mimic_child_info = {}
            return

        self.mimic_parent_id = mimic_parents[0].id

        mimic_children_info = {
            # Robotiq 85
            "right_outer_knuckle_joint": (1.0, 0.0),
            "left_inner_knuckle_joint": (1.0, 0.0),
            "right_inner_knuckle_joint": (1.0, 0.0),
            "left_inner_finger_joint": (-1.0, 0.0),
            "right_inner_finger_joint": (-1.0, 0.0),
            # SusGrip 2F (multiplier, offset)
            "base_slider_l_joint": (0.036, 0.000753),
            "slider_outer_l_joint": (-9.632, 0.609479),
            "finger_outer_l_joint": (9.632, -0.609479),
            "pad_inner_l_joint": (19.264, -1.218957),
            "passive_pad_inner_l_joint": (19.264, -1.218957),
            "finger_inner_l_joint": (-9.632, 0.609479),
            "base_slider_r_joint": (0.036, 0.000753),
            "slider_outer_r_joint": (9.632, -0.609479),
            "finger_outer_r_joint": (9.632, -0.609479),
            "pad_inner_r_joint": (-19.264, 1.218957),
            "passive_pad_inner_r_joint": (-19.264, 1.218957),
            "finger_inner_r_joint": (9.632, -0.609479),
        }

        self.mimic_child_info = {
            j.id: mimic_children_info[j.name]
            for j in self.joints
            if j.name in mimic_children_info and j.id != self.mimic_parent_id
        }

        self.gripper_range = [0.0, 0.085]

        if mimic_parents[0].name == "finger_joint":
            # Create gear constraints for Robotiq 85 (where offsets are 0.0)
            for j_id, (mult, offset) in self.mimic_child_info.items():
                if offset == 0.0:
                    cid = p.createConstraint(
                        self.robot_id,
                        self.mimic_parent_id,
                        self.robot_id,
                        j_id,
                        jointType=p.JOINT_GEAR,
                        jointAxis=[0, 1, 0],
                        parentFramePosition=[0, 0, 0],
                        childFramePosition=[0, 0, 0],
                    )
                    p.changeConstraint(cid, gearRatio=-mult, maxForce=500, erp=1.0)

    def configure_friction(self):
        """Áp dụng ma sát mục tiêu: Ma sát cao (lateralFriction=5.0) cho ngón kẹp và ma sát vừa phải (0.8) cho thân tay máy."""
        num_joints = p.getNumJoints(self.robot_id)
        for link_id in range(-1, num_joints):
            link_name = "base_link" if link_id == -1 else p.getJointInfo(self.robot_id, link_id)[12].decode("utf-8")
            if link_name in HIGH_FRICTION_LINKS or "pad" in link_name:
                p.changeDynamics(
                    self.robot_id,
                    link_id,
                    lateralFriction=5.0,
                    spinningFriction=0.5,
                    frictionAnchor=1,
                )
            else:
                p.changeDynamics(
                    self.robot_id,
                    link_id,
                    lateralFriction=0.8,
                    spinningFriction=0.1,
                )

    def move_gripper(self, open_length: float, force: float = 500.0):
        """Set gripper target opening width."""
        if self.mimic_parent_id is None:
            return

        open_length = float(np.clip(open_length, self.gripper_range[0], self.gripper_range[1]))
        parent_joint = [j for j in self.joints if j.id == self.mimic_parent_id]

        if parent_joint and parent_joint[0].name in ("gripper_joint", "base_slider_l_joint"):
            p.setJointMotorControl2(
                self.robot_id,
                self.mimic_parent_id,
                p.POSITION_CONTROL,
                targetPosition=open_length,
                force=force,
                positionGain=1.0,
                velocityGain=1.0,
            )
            for j_id, (mult, offset) in self.mimic_child_info.items():
                p.setJointMotorControl2(
                    self.robot_id,
                    j_id,
                    p.POSITION_CONTROL,
                    targetPosition=mult * open_length + offset,
                    force=force,
                    positionGain=1.0,
                    velocityGain=1.0,
                )
        else:
            target_pos = (
                0.715 - math.asin((open_length - 0.010) / 0.1143)
                if open_length > 0.01
                else 0.715
            )
            p.setJointMotorControl2(
                self.robot_id,
                self.mimic_parent_id,
                p.POSITION_CONTROL,
                targetPosition=target_pos,
                force=force,
            )

    def close_until_contact(self, target_obj_id: Optional[int] = None) -> bool:
        """Close gripper until contact is made, then apply strong clamping force to maintain grasp."""
        if self.mimic_parent_id is None:
            return False

        pad_names = [
            "sus2f_pad_l_link", "sus2f_pad_r_link",
            "sus2f_passive_pad_l_link", "sus2f_passive_pad_r_link",
            "sus2f_finger_l_link", "sus2f_finger_r_link",
            "robotiq_85_left_finger_pad", "robotiq_85_right_finger_pad",
            "left_inner_finger_pad", "right_inner_finger_pad",
        ]
        pad_idxs = set(self._link_name_to_index[l] for l in pad_names if l in self._link_name_to_index)

        grip_val = self.gripper_range[1]
        max_iters = 30
        step_size = (self.gripper_range[1] - self.gripper_range[0]) / max_iters

        contact_count = 0
        for _ in range(max_iters):
            if target_obj_id is not None:
                cts = p.getContactPoints(bodyA=self.robot_id, bodyB=target_obj_id) + p.getContactPoints(bodyA=target_obj_id, bodyB=self.robot_id)
                finger_cts = [c for c in cts if (c[3] if c[1] == self.robot_id else c[4]) in pad_idxs]
                if len(finger_cts) > 0:
                    contact_count += 1
                    if contact_count >= 2:
                        self.clamped_grip_val = max(self.gripper_range[0], grip_val - 0.005)
                        self.move_gripper(self.clamped_grip_val, force=500.0)
                        for _ in range(5):
                            p.stepSimulation()
                        return True
            if grip_val <= self.gripper_range[0]:
                break
            grip_val -= step_size
            self.move_gripper(grip_val, force=200.0)
            for _ in range(2):
                p.stepSimulation()

        self.clamped_grip_val = max(self.gripper_range[0], grip_val - 0.005)
        return True

    def is_grasping(self, obj_id: Optional[int]) -> bool:
        """
        Kiểm tra xem vật thể (obj_id) có đang thực sự được kẹp giữa 2 ngón kẹp của gripper hay không.
        Yêu cầu có tiếp xúc (contact) ở CẢ 2 PHÍA (bên Trái + bên Phải).
        """
        if obj_id is None:
            return False

        finger_links = [
            "sus2f_pad_l_link", "sus2f_pad_r_link",
            "sus2f_passive_pad_l_link", "sus2f_passive_pad_r_link",
            "sus2f_finger_l_link", "sus2f_finger_r_link",
            "robotiq_85_left_finger_pad", "robotiq_85_right_finger_pad",
            "left_inner_finger_pad", "right_inner_finger_pad",
        ]

        inv_map = {v: k for k, v in self._link_name_to_index.items()}
        contacts = p.getContactPoints(bodyA=self.robot_id, bodyB=obj_id) + p.getContactPoints(bodyA=obj_id, bodyB=self.robot_id)
        contacted_fingers = set()
        for c in contacts:
            r_link_idx = c[3] if c[1] == self.robot_id else c[4]
            if r_link_idx in inv_map:
                link_name = inv_map[r_link_idx]
                if link_name in finger_links:
                    contacted_fingers.add(link_name)

        has_left = any("_l_" in name or "left" in name for name in contacted_fingers)
        has_right = any("_r_" in name or "right" in name for name in contacted_fingers)
        return bool(has_left and has_right)
