"""
collect_data.py

Script thu thập dữ liệu chuyên sâu cho các task cấu hình trong YAML (config/tasks/),
tự động xuất ra định dạng LeRobotDataset để sẵn sàng huấn luyện SmolVLA / VLA.

Cách chạy:
    python scripts/collect_data.py --task pick_cube_to_bowl --episodes 20 --gui
    python scripts/collect_data.py --task lift_cube --episodes 50 --no-gui
    python scripts/collect_data.py --task pick_cube_to_bowl --root ./my_dataset --repo-id local/pick_cube
"""

import argparse
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pybullet as p

# Thêm root dự án vào sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    LeRobotDataset = None

from envs.scene_config import TaskConfig
from envs.base_env import ManipulationEnv

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config" / "tasks"
if not CONFIG_DIR.exists():
    CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs" / "tasks"


# =========================================================
# LeRobot Dataset Schema Builder
# =========================================================
def build_lerobot_dataset(repo_id: str, root: str, cameras: list, fps: int = 10):
    """
    Tự động xây dựng schema LeRobotDataset dựa trên danh sách camera trong task YAML.
    """
    if LeRobotDataset is None:
        sys.exit("[LỖI] Thư viện 'lerobot' chưa được cài đặt. Hãy kích hoạt môi trường 'ur_bullet'.")

    features = {}

    # Thêm từng camera vào schema video
    for cam in cameras:
        cam_key = f"observation.images.{cam.name}"
        features[cam_key] = {
            "dtype": "video",
            "shape": (cam.height, cam.width, 3),
            "names": ["height", "width", "channel"],
        }

    # State robot (7D: 6 khớp arm + 1 khớp gripper)
    features["observation.state"] = {
        "dtype": "float32",
        "shape": (7,),
        "names": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "gripper"],
    }

    # Action (7D: dx, dy, dz, droll, dpitch, dyaw, gripper)
    features["action"] = {
        "dtype": "float32",
        "shape": (7,),
        "names": ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"],
    }

    if Path(root).exists():
        shutil.rmtree(root)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=root,
        fps=fps,
        features=features,
        use_videos=True,
        image_writer_processes=2,
        image_writer_threads=8,
        batch_encoding_size=1,
        streaming_encoding=False,
    )
    return dataset


# =========================================================
# Scripted Expert Policy
# =========================================================
def make_expert_action(env: ManipulationEnv, phase: str) -> np.ndarray:
    """Tạo 7D action chuyên gia cho từng giai đoạn của bài toán."""
    ee_pos, _ = env.get_ee_pose()

    obj_name = "cube" if "cube" in env.object_ids else list(env.object_ids.keys())[0]
    target_name = "bowl" if "bowl" in env.object_ids else None

    obj_pos, _ = p.getBasePositionAndOrientation(env.object_ids[obj_name])
    obj_pos = np.array(obj_pos, dtype=np.float32)

    tgt_pos = None
    if target_name and target_name in env.object_ids:
        tpos, _ = p.getBasePositionAndOrientation(env.object_ids[target_name])
        tgt_pos = np.array(tpos, dtype=np.float32)

    gripper_offset = getattr(env.cfg.robot, "gripper_finger_offset", 0.080)


    approach_height = obj_pos[2] + gripper_offset + 0.10
    pregrasp_height = obj_pos[2] + gripper_offset + 0.04
    grasp_height = obj_pos[2] + gripper_offset
    table_z = 0.65
    lift_height = table_z + gripper_offset + 0.22

    open_gripper = float(env.gripper_range[1])
    close_gripper = float(env.gripper_range[0])


    xy_noise = env.rng.uniform(-0.001, 0.001, size=2).astype(np.float32)

    if phase == "approach":
        target = np.array([obj_pos[0] + xy_noise[0], obj_pos[1] + xy_noise[1], approach_height], dtype=np.float32)
        gripper = open_gripper
    elif phase == "pre_descend":
        target = np.array([obj_pos[0] + xy_noise[0], obj_pos[1] + xy_noise[1], pregrasp_height], dtype=np.float32)
        gripper = open_gripper
    elif phase == "descend":
        target = np.array([obj_pos[0], obj_pos[1], grasp_height], dtype=np.float32)
        gripper = open_gripper
    elif phase in ("close", "hold_close"):
        target = ee_pos.copy()
        gripper = close_gripper
    elif phase == "lift":
        target = np.array([ee_pos[0], ee_pos[1], lift_height], dtype=np.float32)
        gripper = close_gripper
    elif phase == "move_to_target":
        if tgt_pos is not None:
            target = np.array([tgt_pos[0], tgt_pos[1], lift_height], dtype=np.float32)
        else:
            target = np.array([ee_pos[0], ee_pos[1], lift_height], dtype=np.float32)
        gripper = close_gripper
    elif phase == "place":
        if tgt_pos is not None:
            target = np.array([tgt_pos[0], tgt_pos[1], tgt_pos[2] + gripper_offset + 0.08], dtype=np.float32)
        else:
            target = ee_pos.copy()
        gripper = open_gripper
    else:
        target = ee_pos.copy()
        gripper = open_gripper

    step_speed = 0.015 if phase == "lift" else 0.025
    dir_vec = target - ee_pos
    dist = np.linalg.norm(dir_vec)
    if dist > step_speed:
        delta_pos = (dir_vec / dist) * step_speed
    else:
        delta_pos = dir_vec

    return np.array([delta_pos[0], delta_pos[1], delta_pos[2], 0.0, 0.0, 0.0, gripper], dtype=np.float32)



def phase_done(env: ManipulationEnv, phase: str, step_count: int) -> bool:
    """Kiểm tra điều kiện chuyển sang giai đoạn (phase) tiếp theo."""
    ee_pos, _ = env.get_ee_pose()
    obj_name = "cube" if "cube" in env.object_ids else list(env.object_ids.keys())[0]
    obj_pos, _ = p.getBasePositionAndOrientation(env.object_ids[obj_name])
    obj_pos = np.array(obj_pos, dtype=np.float32)

    gripper_offset = getattr(env.cfg.robot, "gripper_finger_offset", 0.080)


    if phase == "approach":
        target = np.array([obj_pos[0], obj_pos[1], obj_pos[2] + gripper_offset + 0.10], dtype=np.float32)
        return np.linalg.norm(ee_pos - target) < 0.008 or step_count >= 20
    elif phase == "pre_descend":
        target = np.array([obj_pos[0], obj_pos[1], obj_pos[2] + gripper_offset + 0.04], dtype=np.float32)
        return np.linalg.norm(ee_pos - target) < 0.005 or step_count >= 20
    elif phase == "descend":
        target = np.array([obj_pos[0], obj_pos[1], obj_pos[2] + gripper_offset], dtype=np.float32)
        return np.linalg.norm(ee_pos - target) < 0.003 or step_count >= 25
    elif phase == "close":
        return env.has_closed_gripper or step_count >= 10
    elif phase == "hold_close":
        return step_count >= 8
    elif phase == "lift":
        lift_target = np.array([ee_pos[0], ee_pos[1], 0.65 + gripper_offset + 0.22], dtype=np.float32)
        return np.linalg.norm(ee_pos - lift_target) < 0.01 or step_count >= 25
    elif phase == "move_to_target":
        target_name = "bowl" if "bowl" in env.object_ids else None
        if target_name and target_name in env.object_ids:
            tpos, _ = p.getBasePositionAndOrientation(env.object_ids[target_name])
            tgt_xy = np.array(tpos[:2], dtype=np.float32)
            return np.linalg.norm(ee_pos[:2] - tgt_xy) < 0.02 or step_count >= 40
        return step_count >= 40
    elif phase == "place":
        return env.is_success() or step_count >= 30

    return False



# =========================================================
# Execution & Collection Loop
# =========================================================
def collect_dataset(
    task_name: str,
    repo_id: str,
    root: str,
    num_episodes: int,
    gui: bool,
    seed: int,
):
    config_path = CONFIG_DIR / f"{task_name}.yaml"
    if not config_path.exists():
        available = [f.stem for f in CONFIG_DIR.glob("*.yaml")]
        sys.exit(f"[LỖI] Không tìm thấy config '{task_name}'. Các task hiện có: {available}")

    task_cfg = TaskConfig.from_yaml(str(config_path))

    root_path = Path(root)
    if root_path.exists():
        print(f"[INFO] Xóa dữ liệu cũ tại thư mục: {root_path.resolve()}")
        shutil.rmtree(root_path)

    env = ManipulationEnv(task_cfg, gui=gui, seed=seed)
    dataset = build_lerobot_dataset(repo_id=repo_id, root=root, cameras=task_cfg.cameras, fps=20)

    saved_count = 0
    dropped_count = 0

    has_bowl = "bowl" in [obj.name for obj in task_cfg.objects]
    phases = ["approach", "pre_descend", "descend", "close", "hold_close", "lift"]
    if has_bowl:
        phases.extend(["move_to_target", "place"])

    grasp_check_phases = {"lift", "move_to_target"}

    try:
        for ep_idx in range(num_episodes):
            obs = env.reset()

            phase_idx = 0
            phase_step_count = 0
            success = False
            grasp_maintained = True
            lost_grasp_count = 0
            obj_name = "cube" if "cube" in env.object_ids else list(env.object_ids.keys())[0]

            for _ in range(task_cfg.episode.max_steps):
                current_phase = phases[phase_idx]
                action = make_expert_action(env, current_phase)

                # Ghi frame vào LeRobotDataset buffer
                frame = {}
                for cam in task_cfg.cameras:
                    cam_key = f"observation.images.{cam.name}"
                    frame[cam_key] = obs["images"][cam.name]

                frame["observation.state"] = obs["joint_positions"]
                frame["action"] = action.astype(np.float32)
                frame["task"] = obs["instruction"]

                dataset.add_frame(frame)

                obs, reward, done, _ = env.apply_action(action)
                phase_step_count += 1

                # Đảm bảo duy trì kẹp giữ vật trong suốt các phase nâng/di chuyển/đặt
                if current_phase in grasp_check_phases:
                    if not env.is_object_grasped(obj_name):
                        lost_grasp_count += 1
                        if lost_grasp_count > 20:
                            grasp_maintained = False
                    else:
                        lost_grasp_count = 0

                if phase_done(env, current_phase, phase_step_count):
                    ee_pos, _ = env.get_ee_pose()
                    obj_pos, _ = p.getBasePositionAndOrientation(env.object_ids[obj_name])
                    print(f"  [DEBUG Phase {current_phase}->Next] Step={phase_step_count} EE_Z={ee_pos[2]:.3f} Obj_Z={obj_pos[2]:.3f}")
                    phase_idx += 1
                    phase_step_count = 0
                    if phase_idx >= len(phases):
                        done = True

                if done:
                    success = env.is_success() and grasp_maintained
                    break

            if success:
                dataset.save_episode()
                saved_count += 1
                print(f"[Episode {ep_idx + 1:03d}/{num_episodes}] SAVED  | Task: '{task_cfg.task_name}' | Instruction: '{obs['instruction']}' (Saved={saved_count}, Dropped={dropped_count})")
            else:
                dataset.clear_episode_buffer(delete_images=True)
                dropped_count += 1
                reason = "grasp lost" if not grasp_maintained else "position check failed"
                print(f"[Episode {ep_idx + 1:03d}/{num_episodes}] DROPPED ({reason}) | Task: '{task_cfg.task_name}' | Instruction: '{obs['instruction']}' (Saved={saved_count}, Dropped={dropped_count})")

    finally:
        dataset.finalize()
        env.close()
        print("\n=========================================================")
        print(f"[HOÀN THÀNH] Đã xuất LeRobotDataset tại '{root}'")
        print(f"Tổng số Episode thành công (SAVED): {saved_count} / {num_episodes}")
        print("=========================================================")


def preview_environment(task_name: str, seed: int = 42, seconds: float = 20.0):
    """Mở giao diện GUI xem trước môi trường và camera mà không ghi dữ liệu."""
    config_path = CONFIG_DIR / f"{task_name}.yaml"
    if not config_path.exists():
        available = [f.stem for f in CONFIG_DIR.glob("*.yaml")]
        sys.exit(f"[LỖI] Không tìm thấy config '{task_name}'. Các task hiện có: {available}")

    task_cfg = TaskConfig.from_yaml(str(config_path))
    env = ManipulationEnv(task_cfg, gui=True, seed=seed)
    try:
        env.reset()
        print(f"[PREVIEW] Đang xem trước task '{task_name}' trong {seconds} giây...")
        for _ in range(int(seconds * 240)):
            p.stepSimulation()
            time.sleep(1.0 / 240.0)
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description="Thu thập LeRobotDataset cho SmolVLA dựa trên Task YAML")
    parser.add_argument("--task", default="lift_cube", help="Tên file YAML trong config/tasks/ (không cần .yaml)")
    parser.add_argument("--preview", action="store_true", help="Mở giao diện xem trước GUI (không ghi dữ liệu)")
    parser.add_argument("--preview-seconds", type=float, default=20.0, help="Thời gian xem trước (giây)")
    parser.add_argument("--episodes", type=int, default=10, help="Số episode thành công cần thu thập")
    parser.add_argument("--root", default=None, help="Đường dẫn thư mục lưu LeRobotDataset (Mặc định: ./datasets/<task>_<YYYYMMDD_HHMMSS>)")
    parser.add_argument("--repo-id", default=None, help="Repo ID LeRobot (Mặc định: local/<task>_<YYYYMMDD_HHMMSS>)")
    parser.add_argument("--gui", action="store_true", default=True, help="Hiển thị cửa sổ GUI PyBullet")
    parser.add_argument("--no-gui", dest="gui", action="store_false", help="Chạy ẩn (HEADLESS - Tốc độ cao)")
    parser.add_argument("--seed", type=int, default=42, help="Seed ngẫu nhiên")
    args = parser.parse_args()

    if args.preview:
        preview_environment(task_name=args.task, seed=args.seed, seconds=args.preview_seconds)
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = args.root if args.root else f"./datasets/{args.task}_{timestamp}"
    repo_id = args.repo_id if args.repo_id else f"local/{args.task}_{timestamp}"

    collect_dataset(
        task_name=args.task,
        repo_id=repo_id,
        root=root,
        num_episodes=args.episodes,
        gui=args.gui,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

