"""
human_motion_env.py

Môi trường PyBullet tích hợp dữ liệu chuyển động người (Text-to-Motion HY-Motion).
Mô phỏng chuỗi hành động "human pick the cup" từ file FBX:
- Tạo không gian 3D gồm Mặt sàn, Bàn, Cốc/Vật thể.
- Tạo mô hình người 3D (Human Skeleton/Body Visuals) chuyển động theo quỹ đạo được nội suy từ HY-Motion.
- Cho phép robot (UR3e/UR5) đồng thời tương tác/quan sát chuyển động người.
- Hỗ trợ Render Camera quan sát góc nhìn toàn cảnh (Overview) và góc nhìn thực tế.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pybullet as p
import pybullet_data

from envs.human_motion_player import HumanMotionPlayer


class HumanMotionEnv:
    def __init__(
        self,
        motion_file: str = "/home/hungdao/ur_ws/src/ur_action_intent/hy_motion/handover01/20260805_102754201_70f7665a_000.gltf",
        gui: bool = True,
        human_origin: Tuple[float, float, float] = (0.38, -0.62, 0.0),
        human_scale: float = 0.008,
        use_mesh_body: bool = True,
        use_builtin_urdf: bool = False,
    ):
        self.gui = gui
        self.motion_file = self._resolve_motion_file(motion_file)
        self.human_origin = human_origin
        self.human_scale = human_scale
        self.use_mesh_body = use_mesh_body
        self.use_builtin_urdf = use_builtin_urdf

        if self.gui:
            self._client = p.connect(p.GUI, options="--width=1600 --height=1000")
        else:
            self._client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        self.table_id = None
        self.cup_id = None
        self.human_player: Optional[HumanMotionPlayer] = None
        self.current_frame = 0

        self.reset()

    def _resolve_motion_file(self, target_path: str) -> str:
        p_path = Path(target_path)
        if p_path.exists():
            return str(p_path)
        # Search hy_motion directory recursively for .gltf or .fbx files
        hy_dir = Path("/home/hungdao/ur_ws/src/ur_action_intent/hy_motion")
        if hy_dir.exists():
            cands = list(hy_dir.rglob("*.gltf")) + list(hy_dir.rglob("*.fbx"))
            if cands:
                return str(sorted(cands)[0])
        return str(p_path)

    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)

        # 1. Tải mặt sàn & Bàn làm việc
        p.loadURDF("plane.urdf")
        self.table_id = p.loadURDF(
            "table/table.urdf",
            [0.38, 0, 0],
            p.getQuaternionFromEuler([0, 0, 0]),
        )

        # 2. Tải vật thể Cốc (Cup / Cube) ứng với nhãn "human pick the cup"
        try:
            self.cup_id = p.loadURDF(
                "cube_small.urdf",
                [0.25, 0.20, 0.65],
                p.getQuaternionFromEuler([0, 0, 0]),
                globalScaling=1.2,
            )
            p.changeVisualShape(self.cup_id, -1, rgbaColor=[1.0, 0.5, 0.1, 1.0])
        except Exception:
            self.cup_id = None

        # 3. Khởi tạo mô hình chuyển động người từ HY-Motion FBX/GLTF với 3D Mesh Body
        self.human_player = HumanMotionPlayer(
            motion_file_path=self.motion_file,
            scale=self.human_scale,
            origin_offset=self.human_origin,
        )
        self.human_player.spawn_in_pybullet(
            joint_radius=0.035,
            use_mesh_body=self.use_mesh_body,
            use_builtin_urdf=self.use_builtin_urdf,
        )


        # 4. Cấu hình Debug Camera hướng vào người & bàn làm việc
        if self.gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=2.2,
                cameraYaw=55,
                cameraPitch=-25,
                cameraTargetPosition=[0.2, 0.2, 0.8],
            )

        self.current_frame = 0
        return self.get_obs()

    def step(self):
        """Tiến lên 1 bước thời gian (1 frame trong dữ liệu chuyển động người)."""
        if self.human_player is not None:
            self.human_player.update(self.current_frame)
            self.current_frame = (self.current_frame + 1) % self.human_player.num_frames

        p.stepSimulation()
        if self.gui:
            time.sleep(1.0 / 30.0)  # Phản hồi theo thời gian thực 30 FPS

        done = (self.current_frame == self.human_player.num_frames - 1)
        return self.get_obs(), 0.0, done, {}

    def get_obs(self) -> Dict[str, np.ndarray]:
        """Trả về thông tin quan sát chuyển động của người và trạng thái môi trường."""
        obs = {
            "current_frame": self.current_frame,
            "total_frames": self.human_player.num_frames if self.human_player else 0,
        }

        if self.human_player:
            obs["human_right_hand_pos"] = self.human_player.get_joint_position("R_Wrist", self.current_frame)
            obs["human_left_hand_pos"] = self.human_player.get_joint_position("L_Wrist", self.current_frame)
            obs["human_head_pos"] = self.human_player.get_joint_position("Head", self.current_frame)
            obs["human_pelvis_pos"] = self.human_player.get_joint_position("Pelvis", self.current_frame)

        if self.cup_id is not None:
            pos, orn = p.getBasePositionAndOrientation(self.cup_id)
            obs["cup_pos"] = np.array(pos, dtype=np.float32)

        return obs

    def render_camera_image(
        self,
        width: int = 640,
        height: int = 480,
        camera_pos: List[float] = [1.4, -1.2, 1.2],
        target_pos: List[float] = [0.38, -0.1, 0.7],
    ) -> np.ndarray:

        """Render ảnh RGB từ góc nhìn camera tổng thể."""
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera_pos,
            cameraTargetPosition=target_pos,
            cameraUpVector=[0, 0, 1],
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60.0,
            aspect=float(width) / float(height),
            nearVal=0.1,
            farVal=5.0,
        )
        renderer = p.ER_BULLET_HARDWARE_OPENGL if self.gui else p.ER_TINY_RENDERER
        _, _, rgb, _, _ = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=renderer,
        )
        return np.reshape(rgb, (height, width, 4))[:, :, :3].astype(np.uint8)

    def close(self):
        p.disconnect(self._client)
