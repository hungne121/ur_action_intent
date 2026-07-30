"""
base_env.py

1 class Environment DUY NHẤT dùng chung cho MỌI task.
Task khác nhau chỉ khác ở file config YAML, không cần viết class riêng
trừ khi success condition quá đặc thù (xem hàm _check_success để mở rộng).

Tích hợp đầy đủ các tính năng đã triển khai trong dự án:
- Tự động biên dịch xacro sang URDF linh hoạt (đọc ROS_SETUP_PATHS từ env/config).
- Hỗ trợ robot UR3e / UR5 với tay kẹp SusGrip 2F hoặc Robotiq 85 (mimic joints).
- Tự động tìm arm joints theo tên chuẩn UR & end-effector link để tính IK.
- Cấu hình lực ma sát cao mục tiêu (HIGH_FRICTION_LINKS) trên ngón kẹp để giữ vật chắc chắn mà không gây sốc vật lý.
- Tô màu chuẩn thực tế (authentic colors) cho robot và gripper.
- Hỗ trợ camera cố định & camera gắn trên tay robot (eye-in-hand dynamic camera).
- Đa dạng không gian action (dict joint target, 7D delta EE pose gồm cả rotation, 6D/7D joint vector).
- Trả về observation phong phú (joint states, EE pose, object poses, RGB images, language instructions).
- Kiểm tra điều kiện hoàn thành (success conditions) và thất bại (failure conditions) chặt chẽ.
"""

import math
import os
import subprocess
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
    from scripts.scene_config import TaskConfig
except ImportError:
    try:
        from envs.scene_config import TaskConfig
    except ImportError:
        from scene_config import TaskConfig


# Cấu hình danh sách đường dẫn ROS setup.bash cho xacro compile (dễ di động giữa các máy)
ROS_SETUP_PATHS = os.environ.get(
    "ROS_SETUP_PATHS",
    "/opt/ros/humble/setup.bash:/home/hungdao/ur_ws/install/setup.bash"
).split(":")

# Danh sách các link ngón kẹp cần ma sát cao để giữ vật chắc chắn
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


def _resolve_urdf_path(urdf_path: str) -> str:
    """
    Resolve URDF / XACRO file path dynamically.
    Compiles .xacro to .urdf if needed.
    """
    path_obj = Path(urdf_path)

    # Check absolute or direct relative path first
    if path_obj.exists():
        target_path = path_obj
    elif (_PROJECT_ROOT / path_obj).exists():
        target_path = _PROJECT_ROOT / path_obj
    elif (_PROJECT_ROOT / "urdf" / path_obj.name).exists():
        target_path = _PROJECT_ROOT / "urdf" / path_obj.name
    elif (_PROJECT_ROOT / "urdf" / "objects" / path_obj.name).exists():
        target_path = _PROJECT_ROOT / "urdf" / "objects" / path_obj.name
    else:
        # Pybullet default asset search
        return urdf_path

    # If it's a xacro file or if there's a corresponding .xacro file to compile
    if target_path.suffix == ".xacro":
        urdf_target = target_path.with_suffix("")
        if not urdf_target.name.endswith(".urdf"):
            urdf_target = target_path.with_suffix(".urdf")
        _compile_xacro_if_needed(target_path, urdf_target)
        return str(urdf_target)

    # Check if a .xacro sibling exists for this urdf file
    xacro_sibling = target_path.with_suffix(".urdf.xacro")
    if not xacro_sibling.exists():
        xacro_sibling = target_path.with_name(target_path.stem + ".xacro")
    if xacro_sibling.exists():
        _compile_xacro_if_needed(xacro_sibling, target_path)

    return str(target_path)


def _compile_xacro_if_needed(xacro_path: Path, urdf_path: Path):
    """Compile XACRO file to URDF if outdated or missing using portable ROS paths."""
    if not urdf_path.exists() or xacro_path.stat().st_mtime > urdf_path.stat().st_mtime:
        print(f"[XACRO] Compiling {xacro_path.name} -> {urdf_path.name}...")
        valid_sources = [f"source '{p}' 2>/dev/null" for p in ROS_SETUP_PATHS if Path(p).exists() or "/opt/ros" in p]
        source_cmds = " && ".join(valid_sources) if valid_sources else ":"
        cmd = f"{source_cmds}; xacro '{xacro_path}' > '{urdf_path}'"
        try:
            subprocess.run(cmd, shell=True, executable="/bin/bash", check=True)
            print("[XACRO] Compilation complete!")
        except Exception as e:
            print(f"[XACRO Warning] Failed to compile xacro: {e}")


class ManipulationEnv:
    def __init__(self, task_config: TaskConfig, gui: bool = True, seed: Optional[int] = None):
        self.cfg = task_config
        self.rng = np.random.default_rng(seed)
        self.gui = gui

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
        self.mimic_parent_id = None
        self.mimic_child_info = {}
        self.gripper_range = [0.0, 0.085]
        self.has_closed_gripper = False
        self.current_step = 0
        self.current_instruction = ""
        self.fixed_ee_euler = [np.pi, 0.0, -np.pi / 2.0]

        self.reset()

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
        self._setup_mimic_joints()
        self._apply_authentic_robot_colors()
        self._configure_friction()

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

    def _setup_mimic_joints(self):
        """Tie child gripper joints to parent joint (Robotiq 85 & SusGrip 2F)."""
        mimic_parents = [j for j in self.joints if j.name in ("finger_joint", "gripper_joint", "base_slider_l_joint")]
        if not mimic_parents:
            self.mimic_parent_id = None
            self.mimic_child_info = {}
            return

        self.mimic_parent_id = mimic_parents[0].id

        # Hệ số (multiplier, offset) khớp mimic được đo/khớp từ mô hình CAD và cơ cấu truyền động vật lý
        # của SusGrip 2F và Robotiq 85 khi chuyển động đồng bộ.
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

    def _configure_friction(self):
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
        for obj in self.cfg.objects:
            pos, yaw = obj.spawn_region.sample_position(self.rng)
            quat = p.getQuaternionFromEuler([0, 0, yaw])
            urdf_resolved = _resolve_urdf_path(obj.urdf)

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
        if self.gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=1.0,
                cameraYaw=50,
                cameraPitch=-35,
                cameraTargetPosition=[0.35, 0, 0.2],
            )

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

    def gripper_close(self, target_obj_name: Optional[str] = None) -> bool:
        """Close gripper until contact is made, then apply strong clamping force to maintain grasp."""
        if self.mimic_parent_id is None:
            return False

        target_obj_id = self.object_ids.get(target_obj_name) if target_obj_name else None
        if target_obj_id is None and self.object_ids:
            target_obj_id = list(self.object_ids.values())[0]

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

    def is_object_grasped(self, obj_name: str, min_contacts: int = 2) -> bool:
        """
        Kiểm tra xem vật thể (obj_name) có đang thực sự được kẹp giữa 2 ngón kẹp của gripper hay không.
        Yêu cầu có tiếp xúc (contact) ở CẢ 2 PHÍA (bên Trái + bên Phải).
        """
        obj_id = self.object_ids.get(obj_name)
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
                        self.move_gripper(getattr(self, "clamped_grip_val", self.gripper_range[0]), force=500.0)

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
        images = {}
        width_default, height_default = 320, 240

        for cam in self.cfg.cameras:
            w = getattr(cam, "width", width_default)
            h = getattr(cam, "height", height_default)
            fov = getattr(cam, "fov", 60.0)

            if getattr(cam, "attach_to_link", None) is not None:
                view_matrix = self._eye_in_hand_view_matrix(cam.attach_to_link)
            else:
                pos = getattr(cam, "position", [0.9, 0.0, 0.5])
                target = getattr(cam, "target", [0.35, 0.0, 0.1])
                view_matrix = p.computeViewMatrix(
                    cameraEyePosition=pos,
                    cameraTargetPosition=target,
                    cameraUpVector=[0, 0, 1],
                )

            proj_matrix = p.computeProjectionMatrixFOV(
                fov=fov,
                aspect=float(w) / float(h),
                nearVal=0.01,
                farVal=3.0,
            )

            renderer = p.ER_BULLET_HARDWARE_OPENGL if self.gui else p.ER_TINY_RENDERER
            _, _, rgb, _, _ = p.getCameraImage(
                width=w,
                height=h,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=renderer,
            )
            rgb = np.reshape(rgb, (h, w, 4))[:, :, :3].astype(np.uint8)
            images[cam.name] = rgb

        return images

    def _eye_in_hand_view_matrix(self, link_name: str):
        """Compute 3D Eye-In-Hand view matrix for end-effector attached camera."""
        idx = (
            self._link_name_to_index.get(link_name)
            or self._joint_name_to_index.get(link_name)
            or self.eef_id
        )

        link_state = p.getLinkState(self.robot_id, idx, computeForwardKinematics=True)
        pos = np.array(link_state[4] if len(link_state) > 4 else link_state[0])
        orn = link_state[5] if len(link_state) > 5 else link_state[1]

        rot_mat = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        forward = rot_mat[:, 2]   # looking direction
        up = rot_mat[:, 2]        # camera up

        cam_eye = pos + forward * 0.04 + rot_mat[:, 1] * 0.1
        cam_target = pos + forward * 0.5
        return p.computeViewMatrix(cam_eye.tolist(), cam_target.tolist(), up.tolist())

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
