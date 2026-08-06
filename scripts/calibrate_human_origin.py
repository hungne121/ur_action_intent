#!/usr/bin/env python3
"""
calibrate_human_origin.py

Tìm tham số (origin, scale) tối ưu cho human_motion trong YAML sao cho
khớp R_Wrist nằm trong workspace thực tế của UR3e khi tay người giơ ra.

UR3e workspace (từ base joint tại z=0.62):
  - Reach radius   : 0.30m – 0.50m (từ base link)
  - z height       : 0.70m – 0.95m (EE có thể với tới)
  - x target range : 0.28m – 0.46m (phía trước bàn)
  - y target range : -0.25m – 0.10m (phía người đứng)

Chạy:
    python scripts/calibrate_human_origin.py
    python scripts/calibrate_human_origin.py --motion-dir hy_motion/handover01
    python scripts/calibrate_human_origin.py --frame-range 50 90
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


# ── UR3e workspace constraints ────────────────────────────────────────────────
UR3E_BASE_POS = np.array([0.0, 0.0, 0.62])  # base_position từ YAML robot config
UR3E_REACH_MIN = 0.28
UR3E_REACH_MAX = 0.50
WORKSPACE_X = (0.28, 0.46)
WORKSPACE_Y = (-0.28, 0.10)
WORKSPACE_Z = (0.70, 0.95)


# ── GLTF FK computation ───────────────────────────────────────────────────────
def load_gltf_motion(gltf_path: str):
    """Load GLTF animation và trả về (nodes, node_name_to_id, parent_map, times, node_anims)."""
    bin_path = Path(gltf_path).with_suffix(".bin")

    with open(gltf_path) as f:
        gltf = json.load(f)
    with open(bin_path, "rb") as f:
        bin_data = f.read()

    def get_accessor_data(acc_id):
        acc = gltf["accessors"][acc_id]
        bv = gltf["bufferViews"][acc["bufferView"]]
        offset = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
        length = acc["count"]
        components = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}[acc["type"]]
        data = np.frombuffer(bin_data, dtype=np.float32, count=length * components, offset=offset)
        return data.reshape((length, components)) if components > 1 else data

    nodes = gltf["nodes"]
    node_name_to_id = {n.get("name", f"Node_{i}"): i for i, n in enumerate(nodes)}
    parent_map = {}
    for i, n in enumerate(nodes):
        for c in n.get("children", []):
            parent_map[c] = i

    anim = gltf["animations"][0]
    times = get_accessor_data(anim["samplers"][0]["input"])
    node_anims = {i: {} for i in range(len(nodes))}
    for ch in anim["channels"]:
        nid = ch["target"]["node"]
        path = ch["target"]["path"]
        sampler = anim["samplers"][ch["sampler"]]
        vals = get_accessor_data(sampler["output"])
        node_anims[nid][path] = vals

    return nodes, node_name_to_id, parent_map, times, node_anims


def compute_fk_trajectory(joint_name: str, nodes, node_name_to_id, parent_map,
                           times, node_anims, scale: float, origin: np.ndarray) -> np.ndarray:
    """Tính Forward Kinematics cho joint_name, trả về trajectory (N, 3) trong PyBullet coords."""
    if joint_name not in node_name_to_id:
        raise ValueError(f"Joint '{joint_name}' không tìm thấy trong GLTF.")
    tid = node_name_to_id[joint_name]

    def chain_of(nid):
        chain = [nid]
        curr = nid
        while curr in parent_map:
            curr = parent_map[curr]
            chain.append(curr)
        chain.reverse()
        return chain

    def local_mat(node_id, f_idx):
        n = nodes[node_id]
        anims = node_anims[node_id]
        t = anims["translation"][f_idx] if "translation" in anims else np.array(n.get("translation", [0, 0, 0]), dtype=np.float32)
        q = anims["rotation"][f_idx] if "rotation" in anims else np.array(n.get("rotation", [0, 0, 0, 1]), dtype=np.float32)
        s = anims["scale"][f_idx] if "scale" in anims else np.array(n.get("scale", [1, 1, 1]), dtype=np.float32)
        mat = np.eye(4)
        mat[:3, :3] = R.from_quat(q).as_matrix() * s
        mat[:3, 3] = t
        return mat

    chain = chain_of(tid)
    traj = np.zeros((len(times), 3), dtype=np.float32)
    for f in range(len(times)):
        M = np.eye(4)
        for nid in chain:
            M = M @ local_mat(nid, f)
        raw = M[:3, 3]
        # FBX (Y-Up) → PyBullet (Z-Up): X→X, Z→Y, Y→Z
        traj[f] = [
            raw[0] * scale + origin[0],
            raw[2] * scale + origin[1],
            raw[1] * scale + origin[2],
        ]
    return traj


# ── Workspace check ───────────────────────────────────────────────────────────
def in_workspace(pos: np.ndarray) -> bool:
    x, y, z = pos
    dist = float(np.linalg.norm(pos - UR3E_BASE_POS))
    return (
        WORKSPACE_X[0] <= x <= WORKSPACE_X[1] and
        WORKSPACE_Y[0] <= y <= WORKSPACE_Y[1] and
        WORKSPACE_Z[0] <= z <= WORKSPACE_Z[1] and
        UR3E_REACH_MIN <= dist <= UR3E_REACH_MAX
    )


def workspace_score(traj: np.ndarray, frame_range: tuple) -> dict:
    """Tính điểm cho trajectory trong khoảng frame_range."""
    start, end = frame_range
    end = min(end, len(traj))
    window = traj[start:end]
    if len(window) == 0:
        return {"score": 0.0, "in_workspace": 0, "total": 0, "mean_pos": [0, 0, 0]}

    in_ws = [in_workspace(p) for p in window]
    mean_pos = window.mean(axis=0)
    mean_dist = float(np.linalg.norm(mean_pos - UR3E_BASE_POS))

    # Score: % frames trong workspace + proximity bonus
    pct = sum(in_ws) / len(in_ws)
    dist_penalty = max(0, mean_dist - UR3E_REACH_MAX) * 5  # penalty nếu quá xa
    score = pct - dist_penalty

    return {
        "score": score,
        "in_workspace_pct": pct * 100,
        "in_workspace": sum(in_ws),
        "total": len(window),
        "mean_pos": mean_pos.tolist(),
        "mean_dist_from_base": mean_dist,
    }


# ── Main sweep ────────────────────────────────────────────────────────────────
def calibrate(motion_dir: str, target_joint: str, frame_range: tuple, verbose: bool):
    motion_path = Path(motion_dir)
    gltf_files = sorted(motion_path.glob("*.gltf"))
    if not gltf_files:
        # Thử tìm FBX và dùng GLTF đã convert
        fbx_files = sorted(motion_path.glob("*.fbx"))
        sys.exit(f"Không tìm thấy GLTF trong {motion_dir}. FBX files: {[f.name for f in fbx_files]}")

    print(f"[INFO] Tìm thấy {len(gltf_files)} GLTF files. Dùng: {gltf_files[0].name}")
    nodes, n2id, par, times, n_anims = load_gltf_motion(str(gltf_files[0]))
    print(f"[INFO] Frames: {len(times)}, Duration: {times[-1]:.2f}s, Target joint: {target_joint}")
    print(f"[INFO] Sweep window: frames {frame_range[0]}–{frame_range[1]}")
    print()

    # Tính raw FK (không offset, scale=1) để hiểu magnitude
    raw_traj = compute_fk_trajectory(target_joint, nodes, n2id, par, times, n_anims,
                                      scale=1.0, origin=np.zeros(3))
    # FBX Y (raw[1]) → pybullet Z; FBX Z (raw[2]) → pybullet Y
    raw_y_range = (raw_traj[:, 1].min(), raw_traj[:, 1].max())
    raw_z_range = (raw_traj[:, 2].min(), raw_traj[:, 2].max())
    raw_x_range = (raw_traj[:, 0].min(), raw_traj[:, 0].max())

    if verbose:
        print(f"  RAW FBX data (trước scale/offset):")
        print(f"    raw_x (→ pb_x): [{raw_x_range[0]:.1f}, {raw_x_range[1]:.1f}]")
        print(f"    raw_y (→ pb_z): [{raw_y_range[0]:.1f}, {raw_y_range[1]:.1f}]")
        print(f"    raw_z (→ pb_y): [{raw_z_range[0]:.1f}, {raw_z_range[1]:.1f}]")
        print()

    # Tính target pybullet coords cho window
    f_start, f_end = frame_range
    f_end = min(f_end, len(times))
    raw_window = raw_traj[f_start:f_end]
    raw_mean_y = raw_window[:, 1].mean()  # → pybullet z
    raw_mean_z = raw_window[:, 2].mean()  # → pybullet y
    raw_mean_x = raw_window[:, 0].mean()  # → pybullet x

    print(f"  Giá trị RAW trung bình tại window [{f_start},{f_end}]:")
    print(f"    raw_x_mean = {raw_mean_x:.2f}  (→ pb_x = raw_x * scale + origin_x)")
    print(f"    raw_y_mean = {raw_mean_y:.2f}  (→ pb_z = raw_y * scale + origin_z)")
    print(f"    raw_z_mean = {raw_mean_z:.2f}  (→ pb_y = raw_z * scale + origin_y)")
    print()

    # Tính origin_z cần thiết cho target pb_z
    scale_candidates = np.arange(0.006, 0.013, 0.001)
    target_pb_z = 0.82    # mục tiêu chiều cao EE handover
    target_pb_y = -0.18   # mục tiêu y: đủ gần robot
    target_pb_x = 0.38    # mục tiêu x: trung tâm workspace

    print("─" * 65)
    print(f"{'Scale':>7} {'origin_x':>10} {'origin_y':>10} {'origin_z':>10}  {'%InWS':>6}  {'DistFromBase':>12}  {'Score':>7}")
    print("─" * 65)

    best = None
    best_score = -999

    for scale in scale_candidates:
        origin_z = target_pb_z - raw_mean_y * scale
        origin_y = target_pb_y - raw_mean_z * scale
        origin_x = target_pb_x - raw_mean_x * scale

        origin = np.array([origin_x, origin_y, origin_z])
        traj = compute_fk_trajectory(target_joint, nodes, n2id, par, times, n_anims, scale, origin)
        result = workspace_score(traj, frame_range)

        print(f"  {scale:.4f}  {origin_x:10.4f}  {origin_y:10.4f}  {origin_z:10.4f}  "
              f"{result['in_workspace_pct']:5.1f}%  {result['mean_dist_from_base']:12.4f}m  {result['score']:7.3f}")

        if result["score"] > best_score:
            best_score = result["score"]
            best = {
                "scale": round(float(scale), 4),
                "origin": [round(origin_x, 4), round(origin_y, 4), round(origin_z, 4)],
                "score": result,
            }

    print("─" * 65)
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║               KẾT QUẢ TỐT NHẤT                         ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  scale  : {best['scale']:<50}║")
    ox, oy, oz = best['origin']
    print(f"║  origin : [{ox}, {oy}, {oz}]")
    print(f"║  %InWorkspace: {best['score']['in_workspace_pct']:.1f}%  ({best['score']['in_workspace']}/{best['score']['total']} frames)")
    mean_pos = best['score']['mean_pos']
    print(f"║  R_Wrist mean pos: x={mean_pos[0]:.3f}, y={mean_pos[1]:.3f}, z={mean_pos[2]:.3f}")
    print(f"║  Dist từ base: {best['score']['mean_dist_from_base']:.4f}m")
    print("╚══════════════════════════════════════════════════════════╝")
    print()
    print("Paste vào handover_cube_to_human.yaml:")
    print(f"  human_motion:")
    print(f"    origin: [{ox}, {oy}, {oz}]")
    print(f"    scale: {best['scale']}")

    return best


def main():
    parser = argparse.ArgumentParser(description="Calibrate human_motion origin và scale cho UR3e workspace")
    parser.add_argument("--motion-dir", default="hy_motion/handover01", help="Thư mục chứa file GLTF/FBX")
    parser.add_argument("--joint", default="R_Wrist", help="Joint target để calibrate")
    parser.add_argument("--frame-range", nargs=2, type=int, default=[55, 85],
                        help="Khoảng frames khi tay người giơ ra (default: 55 85)")
    parser.add_argument("--verbose", action="store_true", help="In chi tiết RAW data")
    args = parser.parse_args()

    calibrate(
        motion_dir=args.motion_dir,
        target_joint=args.joint,
        frame_range=tuple(args.frame_range),
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
