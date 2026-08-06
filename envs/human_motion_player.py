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
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pybullet as p
import pybullet_data
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

# Cấu hình màu sắc & bán kính 3D mesh chuẩn cơ thể người (Skin tone, Shirt, Pants, Shoes)
DEFAULT_SKIN_COLOR = [0.92, 0.75, 0.65, 1.0]
DEFAULT_SHIRT_COLOR = [0.15, 0.28, 0.55, 1.0]
DEFAULT_PANTS_COLOR = [0.20, 0.20, 0.25, 1.0]
DEFAULT_SHOES_COLOR = [0.10, 0.10, 0.12, 1.0]

BONE_PARTS_CONFIG = {
    ("Pelvis", "Spine1"): {"radius": 0.085, "color": DEFAULT_SHIRT_COLOR},
    ("Spine1", "Spine2"): {"radius": 0.090, "color": DEFAULT_SHIRT_COLOR},
    ("Spine2", "Spine3"): {"radius": 0.095, "color": DEFAULT_SHIRT_COLOR},
    ("Spine3", "Neck"): {"radius": 0.050, "color": DEFAULT_SKIN_COLOR},
    ("Neck", "Head"): {"radius": 0.045, "color": DEFAULT_SKIN_COLOR},
    ("Spine3", "L_Collar"): {"radius": 0.045, "color": DEFAULT_SHIRT_COLOR},
    ("L_Collar", "L_Shoulder"): {"radius": 0.045, "color": DEFAULT_SHIRT_COLOR},
    ("L_Shoulder", "L_Elbow"): {"radius": 0.040, "color": DEFAULT_SHIRT_COLOR},
    ("L_Elbow", "L_Wrist"): {"radius": 0.035, "color": DEFAULT_SKIN_COLOR},
    ("Spine3", "R_Collar"): {"radius": 0.045, "color": DEFAULT_SHIRT_COLOR},
    ("R_Collar", "R_Shoulder"): {"radius": 0.045, "color": DEFAULT_SHIRT_COLOR},
    ("R_Shoulder", "R_Elbow"): {"radius": 0.040, "color": DEFAULT_SHIRT_COLOR},
    ("R_Elbow", "R_Wrist"): {"radius": 0.035, "color": DEFAULT_SKIN_COLOR},
    ("Pelvis", "L_Hip"): {"radius": 0.060, "color": DEFAULT_PANTS_COLOR},
    ("L_Hip", "L_Knee"): {"radius": 0.055, "color": DEFAULT_PANTS_COLOR},
    ("L_Knee", "L_Ankle"): {"radius": 0.048, "color": DEFAULT_PANTS_COLOR},
    ("L_Ankle", "L_Foot"): {"radius": 0.040, "color": DEFAULT_SHOES_COLOR},
    ("Pelvis", "R_Hip"): {"radius": 0.060, "color": DEFAULT_PANTS_COLOR},
    ("R_Hip", "R_Knee"): {"radius": 0.055, "color": DEFAULT_PANTS_COLOR},
    ("R_Knee", "R_Ankle"): {"radius": 0.048, "color": DEFAULT_PANTS_COLOR},
    ("R_Ankle", "R_Foot"): {"radius": 0.040, "color": DEFAULT_SHOES_COLOR},
}


def _compute_bone_transform(p0: np.ndarray, p1: np.ndarray) -> Tuple[np.ndarray, List[float], float]:
    """Tính vị trí trung điểm (center), quaternion định hướng (quat) và độ dài (length) cho hình trụ/capsule 3D."""
    p0 = np.asarray(p0, dtype=np.float32)
    p1 = np.asarray(p1, dtype=np.float32)
    center = (p0 + p1) / 2.0
    vec = p1 - p0
    length = float(np.linalg.norm(vec))
    if length < 1e-4:
        return center, [0.0, 0.0, 0.0, 1.0], 1e-4
    dir_vec = vec / length
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    dot = float(np.dot(z_axis, dir_vec))
    if dot > 0.99999:
        quat = [0.0, 0.0, 0.0, 1.0]
    elif dot < -0.99999:
        quat = [1.0, 0.0, 0.0, 0.0]
    else:
        rot_axis = np.cross(z_axis, dir_vec)
        rot_axis = rot_axis / np.linalg.norm(rot_axis)
        angle = float(np.arccos(np.clip(dot, -1.0, 1.0)))
        quat = R.from_rotvec(rot_axis * angle).as_quat().tolist()
    return center, quat, length



class HumanMotionPlayer:
    def __init__(
        self,
        motion_file_path: str,
        scale: float = 0.01,
        origin_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        auto_floor_align: bool = True,
        joint_color: List[float] = [0.1, 0.6, 0.9, 1.0],
        bone_color: List[float] = [0.9, 0.7, 0.2, 1.0],
    ):
        self.motion_file_path = Path(motion_file_path)
        self.scale = scale
        self.origin_offset = np.array(origin_offset, dtype=np.float32)
        self.auto_floor_align = auto_floor_align
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
        if self.auto_floor_align:
            self._apply_floor_align()

    def _ensure_gltf_exists(self) -> Path:
        path = self.motion_file_path
        if path.suffix.lower() == ".gltf":
            return path
        
        gltf_out = path.with_suffix(".gltf")
        if gltf_out.exists():
            return gltf_out

        # Tự động dùng ctypes libassimp để convert FBX sang GLTF
        try:
            import ctypes
            import sys
            lib_path = None
            for p_cand in [
                os.path.join(sys.prefix, "lib", "libassimp.so"),
                os.path.expanduser("~/miniforge3/envs/ur_bullet/lib/libassimp.so"),
                "libassimp.so",
            ]:
                if os.path.exists(p_cand):
                    lib_path = p_cand
                    break
            if lib_path is None:
                lib_path = "libassimp.so"

            lib = ctypes.CDLL(lib_path)
            lib.aiImportFile.argtypes = [ctypes.c_char_p, ctypes.c_uint]
            lib.aiImportFile.restype = ctypes.c_void_p
            lib.aiExportScene.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
            lib.aiExportScene.restype = ctypes.c_int
            lib.aiReleaseImport.argtypes = [ctypes.c_void_p]
            lib.aiReleaseImport.restype = None

            scene = lib.aiImportFile(str(path).encode("utf-8"), 0)
            if scene:
                ret = lib.aiExportScene(scene, b"gltf2", str(gltf_out).encode("utf-8"), 0)
                lib.aiReleaseImport(scene)
                if ret == 0 and gltf_out.exists():
                    print(f"[INFO] Convert thành công FBX -> GLTF qua libassimp: {gltf_out.name}")
                    return gltf_out
        except Exception as e:
            print(f"[WARN] Libassimp conversion via ctypes failed ({e}), attempting assimp CLI...")

        # Fallback search system assimp
        assimp_bin = shutil.which("assimp") or "assimp"
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


    def _apply_floor_align(self):
        """Dịch chuyển toàn bộ trajectories theo Z để chân chạm sàn (z=0)."""
        min_z = float("inf")
        for foot_name in ["R_Foot", "L_Foot"]:
            if foot_name in self.node_name_to_id:
                nid = self.node_name_to_id[foot_name]
                min_z = min(min_z, float(self.joint_trajectories[nid][:, 2].min()))
        if min_z < float("inf") and min_z != 0.0:
            z_shift = -min_z
            for nid in self.joint_trajectories:
                self.joint_trajectories[nid][:, 2] += z_shift

    def get_joint_position(self, joint_name: str, frame_idx: int) -> np.ndarray:

        if joint_name not in self.node_name_to_id:
            raise KeyError(f"Khớp '{joint_name}' không tồn tại trong mô hình.")
        jid = self.node_name_to_id[joint_name]
        f_clamped = int(np.clip(frame_idx, 0, self.num_frames - 1))
        return self.joint_trajectories[jid][f_clamped]

    def spawn_in_pybullet(
        self,
        joint_radius: float = 0.035,
        use_mesh_body: bool = True,
        use_builtin_urdf: bool = False,
        builtin_urdf_path: str = "humanoid/humanoid.urdf",
        use_debug_lines: bool = False,
    ):
        """
        Khởi tạo mô hình cơ thể người 3D trong PyBullet.
        Hỗ trợ:
        - use_mesh_body=True: Dùng 3D Solid Body Mesh (Capsules/Spheres chuẩn tỷ lệ cơ thể người: Torso, Head, Limbs, Color Palette).
        - use_builtin_urdf=True: Load thêm file URDF humanoid built-in chính thức của PyBullet (humanoid/humanoid.urdf).
        - use_debug_lines=True: Tạo các đường debug lines nối các khớp.
        """
        self.use_mesh_body = use_mesh_body
        self.use_builtin_urdf = use_builtin_urdf
        self.sphere_ids = {}
        self.line_ids = {}
        self.mesh_bone_ids = {}
        self.mesh_hand_ids = {}
        self.mesh_head_id = None
        self.builtin_urdf_id = None

        # 1. Nạp mô hình Built-in Humanoid URDF của PyBullet nếu được yêu cầu
        if self.use_builtin_urdf:
            try:
                p.setAdditionalSearchPath(pybullet_data.getDataPath())
                self.builtin_urdf_id = p.loadURDF(
                    builtin_urdf_path,
                    basePosition=self.origin_offset.tolist(),
                    baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                    useFixedBase=True,
                    globalScaling=self.scale * 100.0 if self.scale != 1.0 else 1.0,
                )
                print(f"[INFO] Nạp thành công PyBullet Built-in Humanoid URDF: '{builtin_urdf_path}' (ID: {self.builtin_urdf_id})")
            except Exception as e:
                print(f"[WARN] Không thể nạp Built-in Humanoid URDF ({e}), chuyển sang 3D Body Mesh...")

        # 2. Tạo 3D Mesh Body Geometry cho khung xương người
        if self.use_mesh_body:
            # A. Dựng 3D Mesh Capsules cho các đoạn xương (Bones)
            for parent_name, child_name in KEY_BONES:
                if parent_name in self.node_name_to_id and child_name in self.node_name_to_id:
                    pid = self.node_name_to_id[parent_name]
                    cid = self.node_name_to_id[child_name]
                    p0 = self.joint_trajectories[pid][0]
                    p1 = self.joint_trajectories[cid][0]

                    center, quat, length = _compute_bone_transform(p0, p1)
                    cfg = BONE_PARTS_CONFIG.get((parent_name, child_name), {"radius": 0.04, "color": DEFAULT_SKIN_COLOR})

                    vis_shape = p.createVisualShape(
                        p.GEOM_CAPSULE,
                        radius=cfg["radius"],
                        length=length,
                        rgbaColor=cfg["color"],
                    )
                    body_id = p.createMultiBody(
                        baseMass=0,
                        baseVisualShapeIndex=vis_shape,
                        basePosition=center.tolist(),
                        baseOrientation=quat,
                    )
                    self.mesh_bone_ids[(pid, cid)] = body_id

            # B. Dựng 3D Mesh Head Sphere tại vị trí khớp Đầu
            if "Head" in self.node_name_to_id:
                head_id = self.node_name_to_id["Head"]
                head_pos = self.joint_trajectories[head_id][0]
                head_vis = p.createVisualShape(
                    p.GEOM_SPHERE,
                    radius=0.088,
                    rgbaColor=DEFAULT_SKIN_COLOR,
                )
                self.mesh_head_id = p.createMultiBody(
                    baseMass=0,
                    baseVisualShapeIndex=head_vis,
                    basePosition=head_pos.tolist(),
                )

            # C. Dựng 3D Mesh Hand Spheres tại các khớp Bàn tay (L_Wrist, R_Wrist)
            for hand_name in ["L_Wrist", "R_Wrist"]:
                if hand_name in self.node_name_to_id:
                    hid = self.node_name_to_id[hand_name]
                    h_pos = self.joint_trajectories[hid][0]
                    hand_vis = p.createVisualShape(
                        p.GEOM_SPHERE,
                        radius=0.038,
                        rgbaColor=DEFAULT_SKIN_COLOR,
                    )
                    self.mesh_hand_ids[hid] = p.createMultiBody(
                        baseMass=0,
                        baseVisualShapeIndex=hand_vis,
                        basePosition=h_pos.tolist(),
                    )
        else:
            # Fallback tạo Visual Spheres cho từng khớp đơn lẻ
            sphere_visual_id = p.createVisualShape(
                p.GEOM_SPHERE,
                radius=joint_radius,
                rgbaColor=self.joint_color,
            )
            for name, jid in self.node_name_to_id.items():
                init_pos = self.joint_trajectories[jid][0]
                body_id = p.createMultiBody(
                    baseMass=0,
                    baseVisualShapeIndex=sphere_visual_id,
                    basePosition=init_pos.tolist(),
                )
                self.sphere_ids[jid] = body_id

        # 3. Tạo Debug Lines nếu được kích hoạt
        if use_debug_lines:
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
        """Cập nhật vị trí và định hướng toàn bộ mô hình cơ thể người 3D mesh theo frame_idx."""
        self.current_frame = int(np.clip(frame_idx, 0, self.num_frames - 1))

        # 1. Cập nhật các khối 3D Mesh Capsules cho xương
        if self.use_mesh_body:
            for (pid, cid), body_id in self.mesh_bone_ids.items():
                p0 = self.joint_trajectories[pid][self.current_frame]
                p1 = self.joint_trajectories[cid][self.current_frame]
                center, quat, _ = _compute_bone_transform(p0, p1)
                p.resetBasePositionAndOrientation(body_id, center.tolist(), quat)

            # Cập nhật vị trí Head Sphere
            if self.mesh_head_id is not None and "Head" in self.node_name_to_id:
                head_id = self.node_name_to_id["Head"]
                head_pos = self.joint_trajectories[head_id][self.current_frame]
                p.resetBasePositionAndOrientation(self.mesh_head_id, head_pos.tolist(), [0, 0, 0, 1])

            # Cập nhật vị trí Hand Spheres
            for hid, body_id in self.mesh_hand_ids.items():
                hand_pos = self.joint_trajectories[hid][self.current_frame]
                p.resetBasePositionAndOrientation(body_id, hand_pos.tolist(), [0, 0, 0, 1])

        # 2. Cập nhật Joint Spheres nếu sử dụng
        for jid, body_id in self.sphere_ids.items():
            pos = self.joint_trajectories[jid][self.current_frame]
            p.resetBasePositionAndOrientation(body_id, pos.tolist(), [0, 0, 0, 1])

        # 3. Cập nhật các đường Debug Lines nếu có
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

