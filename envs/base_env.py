"""
base_env.py

1 class Environment DUY NHẤT dùng chung cho MỌI task.
Kế thừa kiến trúc modular trong envs package:
- envs.urdf_utils: Tự động tìm & biên dịch XACRO -> URDF.
- envs.gripper: GripperController quản lý mimic joints, ma sát ngón kẹp & logic grasping.
- envs.camera: CameraManager quản lý rendering fixed & eye-in-hand cameras.
- envs.base_env: CHỈ đóng vai trò Orchestrator quản lý robot kinematics, IK, physics step & task success check.
"""

import os
import time
from collections import namedtuple
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pybullet as p
import pybullet_data

# Project root calculation
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

try:
    from envs.scene_config import TaskConfig
    from envs.urdf_utils import _resolve_urdf_path, ROS_SETUP_PATHS
    from envs.gripper import GripperController, HIGH_FRICTION_LINKS
    from envs.camera import CameraManager
except ImportError:
    from scene_config import TaskConfig
    from urdf_utils import _resolve_urdf_path, ROS_SETUP_PATHS
    from gripper import GripperController, HIGH_FRICTION_LINKS
    from camera import CameraManager


class ManipulationEnv:
    def __init__(self, task_config: TaskConfig, gui: bool = True, seed: Optional[int] = None):
        self.cfg = task_config
        self.rng = np.random.default_rng(seed)
        self.gui = gui
        self.camera_manager = CameraManager(gui=self.gui)

        # Connect to PyBullet physics server
        if self.gui:
            self._client = p.connect(p.GUI, options="--width=1600 --height=1000")
        else:
            self._client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        self.robot_id = None
        self.object_ids: Dict[str, int] = {}
        self._joint_name_to_index: Dict[str, int] = {}
        self._link_name_to_index: Dict[str, int] = {}

        self.joints = []
        self.controllable_joints = []
        self.arm_controllable_joints = []
        self.eef_id = None
        self.gripper: Optional[GripperController] = None
        self.has_closed_gripper = False
        self.current_step = 0
        self.current_instruction = ""
        self.fixed_ee_euler = [np.pi, 0.0, -np.pi / 2.0]

        self.reset()

    # Properties delegating to gripper for 100% backward compatibility
    @property
    def gripper_range(self) -> List[float]:
        return self.gripper.gripper_range if self.gripper else [0.0, 0.085]

    @property
    def mimic_parent_id(self) -> Optional[int]:
        return self.gripper.mimic_parent_id if self.gripper else None

    @property
    def clamped_grip_val(self) -> float:
        return self.gripper.clamped_grip_val if self.gripper else 0.0

    @clamped_grip_val.setter
    def clamped_grip_val(self, val: float):
        if self.gripper:
            self.gripper.clamped_grip_val = val

    # ---------------------------------------------------------- setup ----
    def reset(self):
        p.resetSimulation()
        p.loadURDF("plane.urdf")
        self.table_id = p.loadURDF(
            "table/table.urdf",
            [0.38, 0, 0],
            p.getQuaternionFromEuler([0, 0, 0]),
        )

        self._load_robot()

        self.set_home_pose()
        self._spawn_objects()
        self._setup_debug_camera()

        self.current_step = 0
        self.has_closed_gripper = False

        # Choose a random language instruction variant if available
        if self.cfg.language_instructions:
            self.current_instruction = str(self.rng.choice(self.cfg.language_instructions))
        else:
            self.current_instruction = self.cfg.task_name

        return self._get_obs()

    def _load_robot(self):
        r = self.cfg.robot
        urdf_resolved = _resolve_urdf_path(r.urdf)

        self.robot_id = p.loadURDF(
            urdf_resolved,
            basePosition=r.base_position,
            useFixedBase=r.fixed_base,
        )

        self._parse_joint_info()
        self.gripper = GripperController(
            self.robot_id,
            self.joints,
            self._joint_name_to_index,
            self._link_name_to_index,
        )
        self._apply_authentic_robot_colors()
        self.gripper.configure_friction()

    def _parse_joint_info(self):
        """Parse joint and link info, locate arm joints by standard UR names and find end-effector link."""
        joint_info_tuple = namedtuple(
            "JointInfo",
            [
                "id",
                "name",
                "type",
                "lowerLimit",
                "upperLimit",
                "maxForce",
                "maxVelocity",
                "controllable",
                "linkName",
            ],
        )
        self.joints = []
        self.controllable_joints = []
        self._joint_name_to_index = {}
        self._link_name_to_index = {}

        num_joints = p.getNumJoints(self.robot_id)
        for i in range(num_joints):
            info = p.getJointInfo(self.robot_id, i)
            joint_id = info[0]
            joint_name = info[1].decode("utf-8")
            joint_type = info[2]
            lower = info[8]
            upper = info[9]
            force = info[10]
            vel = info[11]
            link_name = info[12].decode("utf-8")
            controllable = (joint_type != p.JOINT_FIXED)

            self._joint_name_to_index[joint_name] = joint_id
            self._link_name_to_index[link_name] = joint_id

            if controllable:
                self.controllable_joints.append(joint_id)

            self.joints.append(
                joint_info_tuple(
                    joint_id,
                    joint_name,
                    joint_type,
                    lower,
                    upper,
                    force,
                    vel,
                    controllable,
                    link_name,
                )
            )

        # Xác định 6 joint của tay robot theo tên chuẩn UR để tránh phụ thuộc vào thứ tự trong URDF
        arm_joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        found_arm_joints = [
            self._joint_name_to_index[name]
            for name in arm_joint_names
            if name in self._joint_name_to_index
        ]

        if len(found_arm_joints) == 6:
            self.arm_controllable_joints = found_arm_joints
        else:
            # Fallback lấy 6 joint controllable đầu tiên nếu URDF dùng tên khớp tùy biến khác
            self.arm_controllable_joints = self.controllable_joints[:6]

        self.arm_num_dofs = len(self.arm_controllable_joints)

        # Locate end-effector link index
        eef_candidates = [
            "tool0",
            "flange-tool0",
            "wrist_3-flange",
            "ee_fixed_joint",
            "ee_link",
            "susgrip_base_joint",
            "wrist_3_joint",
            "wrist_3_link",
        ]
        self.eef_id = self.arm_controllable_joints[-1] if self.arm_controllable_joints else 0
        for candidate in eef_candidates:
            if candidate in self._joint_name_to_index:
                self.eef_id = self._joint_name_to_index[candidate]
                break
            elif candidate in self._link_name_to_index:
                self.eef_id = self._link_name_to_index[candidate]
                break

        # Setup IK bounds
        self.ik_lower_limits = []
        self.ik_upper_limits = []
        self.ik_joint_ranges = []
        self.ik_rest_poses = []
        arm_home_poses = [0.0, -1.40, 1.45, -1.60, -1.57, 0.0]

        for idx, j_id in enumerate(self.controllable_joints):
            j_info = self.joints[j_id]
            low, high = j_info.lowerLimit, j_info.upperLimit
            if low >= high:
                low, high = -2 * np.pi, 2 * np.pi
            self.ik_lower_limits.append(low)
            self.ik_upper_limits.append(high)
            self.ik_joint_ranges.append(high - low)
            if idx < len(arm_home_poses):
                self.ik_rest_poses.append(arm_home_poses[idx])
            else:
                self.ik_rest_poses.append(0.0)

    def _apply_authentic_robot_colors(self):
        """Apply UR3e / UR5 & SusGrip authentic color palette."""
        ur_silver = [0.82, 0.82, 0.85, 1.0]
        ur_blue = [0.0, 0.48, 0.72, 1.0]
        ur_dark = [0.22, 0.22, 0.25, 1.0]
        grip_black = [0.15, 0.15, 0.17, 1.0]
        grip_silver = [0.85, 0.85, 0.88, 1.0]
        grip_pad = [0.10, 0.10, 0.12, 1.0]

        color_mapping = {
            'base_link_inertia': ur_dark,
            'shoulder_link': ur_blue,
            'upper_arm_link': ur_silver,
            'forearm_link': ur_silver,
            'wrist_1_link': ur_blue,
            'wrist_2_link': ur_silver,
            'wrist_3_link': ur_blue,
            'sus2f_base_link': grip_black,
            'sus2f_slider_l_link': grip_black,
            'sus2f_slider_r_link': grip_black,
            'sus2f_outer_l_link': grip_silver,
            'sus2f_outer_r_link': grip_silver,
            'sus2f_finger_l_link': grip_silver,
            'sus2f_finger_r_link': grip_silver,
            'sus2f_inner_l_link': grip_silver,
            'sus2f_inner_r_link': grip_silver,
            'sus2f_pad_l_link': grip_pad,
            'sus2f_pad_r_link': grip_pad,
            'sus2f_passive_pad_l_link': grip_pad,
            'sus2f_passive_pad_r_link': grip_pad,
        }

        link_map = {}
        for i in range(-1, p.getNumJoints(self.robot_id)):
            if i == -1:
                name = 'base_link'
            else:
                name = p.getJointInfo(self.robot_id, i)[12].decode()
            link_map[name] = i

        for link_name, rgba in color_mapping.items():
            if link_name in link_map:
                p.changeVisualShape(self.robot_id, link_map[link_name], rgbaColor=rgba)

    def _spawn_objects(self):
        self.object_ids = {}
        spawned_positions = []
        for obj in self.cfg.objects:
            urdf_resolved = _resolve_urdf_path(obj.urdf)

            pos = None
            yaw = 0.0
            for _ in range(50):
                cand_pos, cand_yaw = obj.spawn_region.sample_position(self.rng)

                # Check reachability from robot base (0, 0): R between 0.28m and 0.48m
                r_dist = float(np.hypot(cand_pos[0], cand_pos[1]))
                if not (0.28 <= r_dist <= 0.48):
                    continue

                # Check minimum separation from already spawned objects (>= 0.14m)
                valid = True
                for prev_pos in spawned_positions:
                    dist_prev = float(np.hypot(cand_pos[0] - prev_pos[0], cand_pos[1] - prev_pos[1]))
                    if dist_prev < 0.14:
                        valid = False
                        break

                if valid:
                    pos = cand_pos
                    yaw = cand_yaw
                    break

            if pos is None:
                pos, yaw = obj.spawn_region.sample_position(self.rng)

            spawned_positions.append(pos)
            quat = p.getQuaternionFromEuler([0, 0, yaw])

            body_id = p.loadURDF(
                urdf_resolved,
                basePosition=pos,
                baseOrientation=quat,
                useFixedBase=obj.fixed_base,
            )
            self.object_ids[obj.name] = body_id

            # Apply high friction to spawned objects
            p.changeDynamics(
                body_id,
                -1,
                lateralFriction=3.0,
                spinningFriction=0.1,
                frictionAnchor=1,
            )

    def _setup_debug_camera(self):
        self.camera_manager.setup_debug_camera()

    # -------------------------------------------------------- control helpers ----
    def set_home_pose(self):
        """Reset arm joints to initial home pose."""
        home_positions = getattr(self.cfg.robot, "home_positions", None) or {
            0: 0.0,
            1: -1.40,
            2: 1.45,
            3: -1.60,
            4: -1.57,
            5: 0.0,
        }
        for idx, target in home_positions.items():
            if idx < len(self.arm_controllable_joints):
                j_id = self.arm_controllable_joints[idx]
                p.resetJointState(self.robot_id, j_id, targetValue=float(target))

    def move_arm_ik_absolute(self, target_pos: List[float], target_orn: Optional[List[float]] = None):
        """Control robot arm to target position using Inverse Kinematics."""
        if target_orn is None:
            target_orn = p.getQuaternionFromEuler(self.fixed_ee_euler)

        joint_poses = p.calculateInverseKinematics(
            self.robot_id,
            self.eef_id,
            target_pos,
            target_orn,
            restPoses=self.ik_rest_poses,
            maxNumIterations=500,
            residualThreshold=1e-5,
        )

        for i, j_id in enumerate(self.arm_controllable_joints):
            p.setJointMotorControl2(
                self.robot_id,
                j_id,
                p.POSITION_CONTROL,
                joint_poses[i],
                force=500,
                maxVelocity=3.0,
            )

    def move_gripper(self, open_length: float, force: float = 500.0):
        """Set gripper target opening width."""
        if self.gripper:
            self.gripper.move_gripper(open_length, force=force)

    def gripper_close(self, target_obj_name: Optional[str] = None) -> bool:
        """Close gripper until contact is made, then apply strong clamping force to maintain grasp."""
        if not self.gripper:
            return False
        target_obj_id = self.object_ids.get(target_obj_name) if target_obj_name else None
        if target_obj_id is None and self.object_ids:
            target_obj_id = list(self.object_ids.values())[0]
        return self.gripper.close_until_contact(target_obj_id)

    def is_object_grasped(self, obj_name: str, min_contacts: int = 2) -> bool:
        """
        Kiểm tra xem vật thể (obj_name) có đang thực sự được kẹp giữa 2 ngón kẹp của gripper hay không.
        Yêu cầu có tiếp xúc (contact) ở CẢ 2 PHÍA (bên Trái + bên Phải).
        """
        if not self.gripper:
            return False
        obj_id = self.object_ids.get(obj_name)
        return self.gripper.is_grasping(obj_id)

    def get_ee_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return end-effector world position and quaternion orientation."""
        state = p.getLinkState(self.robot_id, self.eef_id, computeForwardKinematics=True)
        pos = np.array(state[4] if len(state) > 4 else state[0], dtype=np.float32)
        orn = np.array(state[5] if len(state) > 5 else state[1], dtype=np.float32)
        return pos, orn

    # -------------------------------------------------------- stepping ----
    def step(self, action: Union[dict, list, np.ndarray]):
        """
        Supports multiple action space formats:
        1. Dict format: {joint_name: target_position} or {"joint_indices": [...], "targets": [...]}
        2. 7D EE delta pose array: [dx, dy, dz, droll, dpitch, dyaw, gripper]
        3. 6D/7D joint position array: [q0, q1, q2, q3, q4, q5, gripper]
        """
        if isinstance(action, dict):
            if "joint_indices" in action and "targets" in action:
                for idx, target in zip(action["joint_indices"], action["targets"]):
                    p.setJointMotorControl2(
                        self.robot_id, idx, p.POSITION_CONTROL,
                        targetPosition=target, force=200,
                    )
            else:
                for joint_name, target in action.items():
                    idx = self._joint_name_to_index.get(joint_name)
                    if idx is not None:
                        p.setJointMotorControl2(
                            self.robot_id, idx, p.POSITION_CONTROL,
                            targetPosition=target, force=200,
                        )
        elif isinstance(action, (list, np.ndarray)):
            action_arr = np.asarray(action, dtype=np.float32)
            if len(action_arr) == 7:
                # EE Delta Action: [dx, dy, dz, droll, dpitch, dyaw, gripper]
                dx, dy, dz, droll, dpitch, dyaw, gripper_cmd = action_arr
                ee_pos, ee_orn = self.get_ee_pose()
                target_pos = (ee_pos + np.array([dx, dy, dz])).tolist()

                # Áp dụng orientation delta nếu droll, dpitch, dyaw khác 0
                if droll == 0.0 and dpitch == 0.0 and dyaw == 0.0:
                    target_orn = p.getQuaternionFromEuler(self.fixed_ee_euler)
                else:
                    curr_euler = p.getEulerFromQuaternion(ee_orn)
                    target_euler = [curr_euler[0] + droll, curr_euler[1] + dpitch, curr_euler[2] + dyaw]
                    target_orn = p.getQuaternionFromEuler(target_euler)

                self.move_arm_ik_absolute(target_pos, target_orn)

                if gripper_cmd > 0.02:
                    self.move_gripper(self.gripper_range[1])
                    self.has_closed_gripper = False
                else:
                    if not self.has_closed_gripper:
                        self.has_closed_gripper = self.gripper_close()
                    else:
                        self.move_gripper(self.clamped_grip_val, force=500.0)

            elif len(action_arr) in (6, 7):
                # Joint angles direct control: [q0..q5, (gripper)]
                for i, target in enumerate(action_arr[:len(self.arm_controllable_joints)]):
                    j_id = self.arm_controllable_joints[i]
                    p.setJointMotorControl2(
                        self.robot_id, j_id, p.POSITION_CONTROL,
                        targetPosition=float(target), force=200,
                    )
                if len(action_arr) == 7:
                    self.move_gripper(float(action_arr[6]))

        substeps = max(1, int(240 / getattr(self.cfg.episode, "control_hz", 20)))
        for _ in range(substeps):
            p.stepSimulation()
            if self.gui:
                time.sleep(1.0 / 240.0)

        self.current_step += 1
        obs = self._get_obs()
        success = self._check_success()
        done = success or self.is_failure() or (self.current_step >= self.cfg.episode.max_steps)
        return obs, success, done

    def apply_action(self, action: Union[dict, list, np.ndarray]):
        """Gym-style return alias: (obs, reward, done, info)."""
        obs, success, done = self.step(action)
        reward = 1.0 if success else 0.0
        info = {"is_success": success}
        return obs, reward, done, info

    # -------------------------------------------------------- observation ----
    def _get_obs(self):
        joint_states = {
            name: p.getJointState(self.robot_id, idx)[0]
            for name, idx in self._joint_name_to_index.items()
        }
        object_poses = {
            name: p.getBasePositionAndOrientation(bid)
            for name, bid in self.object_ids.items()
        }
        ee_pos, ee_orn = self.get_ee_pose()
        images = self._render_cameras()

        joint_pos_array = np.array(
            [p.getJointState(self.robot_id, j)[0] for j in self.arm_controllable_joints]
            + ([p.getJointState(self.robot_id, self.mimic_parent_id)[0]] if self.mimic_parent_id is not None else []),
            dtype=np.float32,
        )

        obs = {
            "joint_states": joint_states,
            "joint_positions": joint_pos_array,
            "ee_pos": ee_pos,
            "ee_orn": ee_orn,
            "object_poses": object_poses,
            "object_positions": {k: np.array(v[0], dtype=np.float32) for k, v in object_poses.items()},
            "images": images,
            "instruction": self.current_instruction,
            "task_en": self.current_instruction,
        }

        # Expose individual camera image keys by camera name and by camera index
        for cam_name, img in images.items():
            obs[f"image_{cam_name}"] = img

        img_list = list(images.values())
        for i, img in enumerate(img_list, start=1):
            obs[f"image_camera{i}"] = img

        return obs

    def get_obs(self):
        """Public alias for _get_obs()."""
        return self._get_obs()

    def _render_cameras(self) -> Dict[str, np.ndarray]:
        return self.camera_manager.render_cameras(
            self.cfg.cameras,
            self.robot_id,
            self._link_name_to_index,
            self._joint_name_to_index,
            self.eef_id,
        )

    # -------------------------------------------------------- success & failure ----
    def _check_success(self) -> bool:
        cond = self.cfg.success_condition
        if cond.type == "object_inside":
            obj_id = self.object_ids.get(cond.object)
            tgt_id = self.object_ids.get(cond.target)
            if obj_id is not None and tgt_id is not None:
                obj_pos, _ = p.getBasePositionAndOrientation(obj_id)
                tgt_pos, _ = p.getBasePositionAndOrientation(tgt_id)
                dist_xy = np.linalg.norm(np.array(obj_pos[:2]) - np.array(tgt_pos[:2]))
                thresh = cond.threshold if cond.threshold is not None else 0.08
                height_ok = abs(obj_pos[2] - tgt_pos[2]) < thresh
                return bool(dist_xy < 0.08 and height_ok)

        elif cond.type in ("object_height_above", "height_above"):
            obj_id = self.object_ids.get(cond.object)
            if obj_id is not None:
                obj_pos, _ = p.getBasePositionAndOrientation(obj_id)
                return bool(obj_pos[2] > (cond.threshold or 0.20))

        elif cond.type == "distance_below":
            obj_id = self.object_ids.get(cond.object)
            tgt_id = self.object_ids.get(cond.target)
            if obj_id is not None and tgt_id is not None:
                a, _ = p.getBasePositionAndOrientation(obj_id)
                b, _ = p.getBasePositionAndOrientation(tgt_id)
                thresh = cond.threshold if cond.threshold is not None else 0.05
                return bool(np.linalg.norm(np.array(a) - np.array(b)) < thresh)

        else:
            raise ValueError(
                f"Không hỗ trợ success_condition.type='{cond.type}'. "
                "Các loại hợp lệ: 'object_inside', 'object_height_above', 'height_above', 'distance_below'."
            )

        return False

    def is_success(self) -> bool:
        return self._check_success()

    def is_failure(self) -> bool:
        for obj_id in self.object_ids.values():
            pos, _ = p.getBasePositionAndOrientation(obj_id)
            if pos[2] < -0.1:
                return True
        return False

    def close(self):
        if getattr(self, "_client", None) is not None:
            p.disconnect(self._client)
            self._client = None

    def disconnect(self):
        self.close()


# Alias for backward compatibility
BaseEnv = ManipulationEnv
