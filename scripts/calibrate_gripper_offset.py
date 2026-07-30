"""
calibrate_gripper_offset.py

Đo khoảng cách chính xác từ ee_pos (link tool0/flange) tới tâm 2 má kẹp thật của SusGrip 2F / Robotiq 85
khi ở trạng thái đóng/gắp thật trong PyBullet.
"""

import sys
from pathlib import Path
import numpy as np
import pybullet as p

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from scripts.scene_config import TaskConfig
    from scripts.base_env import ManipulationEnv
except ImportError:
    from scene_config import TaskConfig
    from base_env import ManipulationEnv


def calibrate_offset(task_name: str = "lift_cube"):
    config_path = _PROJECT_ROOT / "config" / "tasks" / f"{task_name}.yaml"
    if not config_path.exists():
        print(f"[LỖI] Không tìm thấy config {config_path}")
        return

    task_cfg = TaskConfig.from_yaml(str(config_path))
    env = ManipulationEnv(task_cfg, gui=False, seed=42)

    # Đóng gripper hoàn toàn giống trạng thái khi kẹp thật
    env.move_gripper(env.gripper_range[0])
    for _ in range(50):
        p.stepSimulation()

    ee_pos, _ = env.get_ee_pose()
    pad_link_names = [
        "sus2f_pad_l_link",
        "sus2f_pad_r_link",
        "sus2f_passive_pad_l_link",
        "sus2f_passive_pad_r_link",
        "robotiq_85_left_finger_pad",
        "robotiq_85_right_finger_pad",
    ]

    found_offsets = []
    print("\n" + "=" * 60)
    print(" [CALIBRATION] KẾT QUẢ ĐO GRIPPER OFFSET")
    print(f" End-Effector (tool0) EE Z = {ee_pos[2]:.4f} m")
    print("-" * 60)

    for name in pad_link_names:
        if name in env._link_name_to_index:
            idx = env._link_name_to_index[name]
            pad_state = p.getLinkState(env.robot_id, idx, computeForwardKinematics=True)
            pad_pos = pad_state[4] if len(pad_state) > 4 else pad_state[0]
            offset_z = ee_pos[2] - pad_pos[2]
            found_offsets.append(offset_z)
            print(f" -> Link '{name}': pad_z={pad_pos[2]:.4f}m | Offset Z = {offset_z:.4f} m ({offset_z*100:.2f} cm)")

    if found_offsets:
        avg_offset = float(np.mean(found_offsets))
        print("=" * 60)
        print(f" => GRIPPER FINGER OFFSET KHUYÊN DÙNG: {avg_offset:.4f} m ({avg_offset*100:.2f} cm)")
        print("=" * 60 + "\n")
    else:
        print("[WARNING] Không tìm thấy link má kẹp nào phù hợp trong URDF!")

    env.close()

if __name__ == "__main__":
    calibrate_offset()
