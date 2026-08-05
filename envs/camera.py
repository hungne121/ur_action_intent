"""
camera.py

Quản lý toàn bộ hệ thống camera (CameraManager):
- Cấu hình PyBullet debug visualizer camera.
- Render ảnh RGB từ camera cố định và camera eye-in-hand (gắn trên tay robot).
- Tính toán matrix góc nhìn (view matrix) cho eye-in-hand camera theo vị trí 3D thực tế của link.
"""

from typing import Dict, List, Optional
import numpy as np
import pybullet as p


class CameraManager:
    def __init__(self, gui: bool = True):
        self.gui = gui

    def setup_debug_camera(self):
        """Cấu hình góc nhìn debug camera trong GUI."""
        if self.gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=1.0,
                cameraYaw=50,
                cameraPitch=-35,
                cameraTargetPosition=[0.35, 0, 0.2],
            )

    def render_cameras(
        self,
        camera_configs: List,
        robot_id: int,
        link_name_to_index: Dict[str, int],
        joint_name_to_index: Dict[str, int],
        eef_id: int,
    ) -> Dict[str, np.ndarray]:
        """Render RGB images for all configured cameras."""
        images = {}
        width_default, height_default = 320, 240

        for cam in camera_configs:
            w = getattr(cam, "width", width_default)
            h = getattr(cam, "height", height_default)
            fov = getattr(cam, "fov", 60.0)

            if getattr(cam, "attach_to_link", None) is not None:
                view_matrix = self.eye_in_hand_view_matrix(
                    robot_id, cam.attach_to_link, link_name_to_index, joint_name_to_index, eef_id
                )
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

    def eye_in_hand_view_matrix(
        self,
        robot_id: int,
        link_name: str,
        link_name_to_index: Dict[str, int],
        joint_name_to_index: Dict[str, int],
        eef_id: int,
    ):
        """Compute 3D Eye-In-Hand view matrix for end-effector attached camera."""
        idx = (
            link_name_to_index.get(link_name)
            or joint_name_to_index.get(link_name)
            or eef_id
        )

        link_state = p.getLinkState(robot_id, idx, computeForwardKinematics=True)
        pos = np.array(link_state[4] if len(link_state) > 4 else link_state[0])
        orn = link_state[5] if len(link_state) > 5 else link_state[1]

        rot_mat = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        forward = rot_mat[:, 2]   # looking direction
        up = rot_mat[:, 2]        # camera up

        cam_eye = pos + forward * 0.04 + rot_mat[:, 1] * 0.1
        cam_target = pos + forward * 0.5
        return p.computeViewMatrix(cam_eye.tolist(), cam_target.tolist(), up.tolist())
