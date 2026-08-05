"""
human_motion_player.py

Class load và phát lại chuyển động người (Human Motion) từ file FBX/GLTF (ví dụ text-to-motion HY-Motion) trong PyBullet.
Hỗ trợ:
- Tự động chuyển đổi FBX sang GLTF qua assimp (nếu cần).
- Tính Forward Kinematics (FK) khớp theo thời gian (300 khung hình).
- Tạo và cập nhật mô hình cơ thể người (Visual Joints & Bone Links) trong PyBullet.
"""

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation as R

# Danh sách khớp chính để vẽ khung xương người chuẩn SMPL
KEY_BONES = [
    ("Pelvis", "Spine1"),
    ("Spine1", "Spine2"),
    ("Spine2", "Spine3"),
    ("Spine3", "Neck"),
    ("Neck", "Head"),
    ("Spine3", "L_Collar"),
    ("L_Collar", "L_Shoulder"),
    ("L_Shoulder", "L_Elbow"),
    ("L_Elbow", "L_Wrist"),
    ("Spine3", "R_Collar"),
    ("R_Collar", "R_Shoulder"),
    ("R_Shoulder", "R_Elbow"),
    ("R_Elbow", "R_Wrist"),
    ("Pelvis", "L_Hip"),
    ("L_Hip", "L_Knee"),
    ("L_Knee", "L_Ankle"),
    ("L_Ankle", "L_Foot"),
    ("Pelvis", "R_Hip"),
    ("R_Hip", "R_Knee"),
    ("R_Knee", "R_Ankle"),
    ("R_Ankle", "R_Foot"),
]


class HumanMotionPlayer:
    def __init__(
        self,
        motion_file_path: str,
        scale: float = 0.01,
        origin_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        joint_color: List[float] = [0.1, 0.6, 0.9, 1.0], # Blue cyan
        bone_color: List[float] = [0.9, 0.7, 0.2, 1.0],  # Warm gold
    ):
        self.motion_file_path = Path(motion_file_path)
        self.scale = scale
        self.origin_offset = np.array(origin_offset, dtype=np.float32)
        self.joint_color = joint_color
        self.bone_color = bone_color

        self.gltf_path = self._ensure_gltf_exists()
        self.nodes = []
        self.parent_map = {}
        self.node_name_to_id = {}
        self.node_id_to_name = {}
        self.times = np.array([])
        self.num_frames = 0
        self.joint_trajectories: Dict[int, np.ndarray] = {}

        self.sphere_ids: Dict[int, int] = {}
        self.line_ids: Dict[Tuple[int, int], int] = {}

        self.current_frame = 0
        self._load_gltf_and_compute_fk()

    def _ensure_gltf_exists(self) -> Path:
        path = self.motion_file_path
        if path.suffix.lower() == ".gltf":
            return path
        
        gltf_out = path.with_suffix(".gltf")
        if gltf_out.exists():
            return gltf_out

        # Tự động dùng assimp CLI để convert FBX sang GLTF
        assimp_bin = os.path.expanduser("~/miniforge3/envs/lerobot/bin/assimp")
        if not os.path.exists(assimp_bin):
            # Fallback search system assimp
            assimp_bin = "assimp"

        cmd = [assimp_bin, "export", str(path), str(gltf_out)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"Lỗi convert FBX sang GLTF: {res.stderr}")
        
        return gltf_out

    def _load_gltf_and_compute_fk(self):
        with open(self.gltf_path, "r") as f:
            gltf = json.load(f)

        bin_path = self.gltf_path.with_suffix(".bin")
        with open(bin_path, "rb") as f:
            bin_data = f.read()

        def get_accessor_data(acc_id):
            acc = gltf["accessors"][acc_id]
            bv = gltf["bufferViews"][acc["bufferView"]]
            offset = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
            length = acc["count"]
            type_str = acc["type"]
            components = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}[type_str]
            data = np.frombuffer(bin_data, dtype=np.float32, count=length * components, offset=offset)
            if components > 1:
                data = data.reshape((length, components))
            return data

        self.nodes = gltf["nodes"]
        self.parent_map = {}
        self.node_name_to_id = {}
        self.node_id_to_name = {}

        for i, n in enumerate(self.nodes):
            name = n.get("name", f"Node_{i}")
            self.node_name_to_id[name] = i
            self.node_id_to_name[i] = name
            for c in n.get("children", []):
                self.parent_map[c] = i

        anim = gltf["animations"][0]
        self.times = get_accessor_data(anim["samplers"][0]["input"])
        self.num_frames = len(self.times)

        # Trích xuất channel chuyển động của từng khớp
        node_anims = {i: {} for i in range(len(self.nodes))}
        for ch in anim["channels"]:
            node_id = ch["target"]["node"]
            path = ch["target"]["path"]
            sampler = anim["samplers"][ch["sampler"]]
            vals = get_accessor_data(sampler["output"])
            node_anims[node_id][path] = vals

        # Tính Forward Kinematics cho tất cả các frame
        self.joint_trajectories = {i: np.zeros((self.num_frames, 3), dtype=np.float32) for i in range(len(self.nodes))}

        for f_idx in range(self.num_frames):
            world_matrices = {}

            def get_local_mat(node_id):
                n = self.nodes[node_id]
                anims = node_anims[node_id]

                if "translation" in anims:
                    t = anims["translation"][f_idx]
                else:
                    t = np.array(n.get("translation", [0, 0, 0]), dtype=np.float32)

                if "rotation" in anims:
                    q = anims["rotation"][f_idx]
                else:
                    q = np.array(n.get("rotation", [0, 0, 0, 1]), dtype=np.float32)

                if "scale" in anims:
                    s = anims["scale"][f_idx]
                else:
                    s = np.array(n.get("scale", [1, 1, 1]), dtype=np.float32)

                rot_mat = R.from_quat(q).as_matrix()
                mat = np.eye(4)
                mat[:3, :3] = rot_mat * s
                mat[:3, 3] = t
                return mat

            for i in range(len(self.nodes)):
                chain = [i]
                curr = i
                while curr in self.parent_map:
                    curr = self.parent_map[curr]
                    chain.append(curr)
                chain.reverse()

                M = np.eye(4)
                for nid in chain:
                    M = M @ get_local_mat(nid)
                
                # Conversion từ FBX (Y-Up) sang PyBullet (Z-Up):
                # PyBullet_X = FBX_X * scale + origin_x
                # PyBullet_Y = FBX_Z * scale + origin_y
                # PyBullet_Z = FBX_Y * scale + origin_z
                raw_pos = M[:3, 3]
                px = raw_pos[0] * self.scale + self.origin_offset[0]
                py = raw_pos[2] * self.scale + self.origin_offset[1]
                pz = raw_pos[1] * self.scale + self.origin_offset[2]
                self.joint_trajectories[i][f_idx] = np.array([px, py, pz], dtype=np.float32)


    def get_joint_position(self, joint_name: str, frame_idx: int) -> np.ndarray:
        if joint_name not in self.node_name_to_id:
            raise KeyError(f"Khớp '{joint_name}' không tồn tại trong mô hình.")
        jid = self.node_name_to_id[joint_name]
        f_clamped = int(np.clip(frame_idx, 0, self.num_frames - 1))
        return self.joint_trajectories[jid][f_clamped]

    def spawn_in_pybullet(self, joint_radius: float = 0.03):
        """Khởi tạo mô hình visual khớp & xương người trong PyBullet."""
        self.sphere_ids = {}
        self.line_ids = {}

        # Tạo visual spheres cho các khớp chính
        sphere_visual_id = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=joint_radius,
            rgbaColor=self.joint_color,
        )

        for name, jid in self.node_name_to_id.items():
            # Chỉ tạo sphere cho các khớp thuộc cây cơ thể (có vị trí xác định)
            init_pos = self.joint_trajectories[jid][0]
            body_id = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=sphere_visual_id,
                basePosition=init_pos.tolist(),
            )
            self.sphere_ids[jid] = body_id

        # Khởi tạo các đoạn liên kết xương bằng debug lines
        for parent_name, child_name in KEY_BONES:
            if parent_name in self.node_name_to_id and child_name in self.node_name_to_id:
                pid = self.node_name_to_id[parent_name]
                cid = self.node_name_to_id[child_name]
                p0 = self.joint_trajectories[pid][0]
                p1 = self.joint_trajectories[cid][0]

                line_id = p.addUserDebugLine(
                    lineFromXYZ=p0.tolist(),
                    lineToXYZ=p1.tolist(),
                    lineColorRGB=self.bone_color[:3],
                    lineWidth=4.0,
                )
                self.line_ids[(pid, cid)] = line_id

    def update(self, frame_idx: int):
        """Cập nhật vị trí toàn bộ khung xương theo frame_idx."""
        self.current_frame = int(np.clip(frame_idx, 0, self.num_frames - 1))

        # Cập nhật vị trí từng khớp sphere
        for jid, body_id in self.sphere_ids.items():
            pos = self.joint_trajectories[jid][self.current_frame]
            p.resetBasePositionAndOrientation(body_id, pos.tolist(), [0, 0, 0, 1])

        # Cập nhật các đường xương bone lines
        for (pid, cid), line_id in self.line_ids.items():
            p0 = self.joint_trajectories[pid][self.current_frame]
            p1 = self.joint_trajectories[cid][self.current_frame]
            p.addUserDebugLine(
                lineFromXYZ=p0.tolist(),
                lineToXYZ=p1.tolist(),
                lineColorRGB=self.bone_color[:3],
                lineWidth=4.0,
                replaceItemUniqueId=line_id,
            )

    def step(self):
        """Tiến lên 1 khung hình tiếp theo."""
        next_frame = (self.current_frame + 1) % self.num_frames
        self.update(next_frame)
