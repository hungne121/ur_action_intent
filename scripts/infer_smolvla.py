#!/usr/bin/env python3
"""
infer_smolvla.py

Script suy luận (Inference) riêng biệt cho mô hình SmolVLA (Vision-Language-Action).
Hỗ trợ kiểm tra dự đoán action 7D từ quan sát camera + trạng thái robot + câu lệnh prompt.

Sử dụng:
    1. Suy luận 1 bước đơn (Quick Test):
        python scripts/infer_smolvla.py --task handover_cube_to_human --dummy

    2. Suy luận vòng lặp trực quan (Interactive GUI Loop):
        python scripts/infer_smolvla.py --task handover_cube_to_human --dummy --gui --loop --steps 50

    3. Suy luận với Checkpoint SmolVLA thực tế:
        python scripts/infer_smolvla.py --task handover_cube_to_human --policy lerobot/smolvla_checkpoints --gui
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from envs.scene_config import TaskConfig
    from envs.base_env import ManipulationEnv
except ImportError:
    from scene_config import TaskConfig
    from base_env import ManipulationEnv

try:
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False


class DummySmolVLAPolicy:
    """Mock Policy phục vụ kiểm thử suy luận khi chưa có weights thật."""
    def __init__(self, action_dim: int = 7):
        self.action_dim = action_dim

    def reset(self):
        pass

    def select_action(self, batch: Dict[str, Any]) -> torch.Tensor:
        delta_pos = np.random.uniform(-0.015, 0.015, size=3).astype(np.float32)
        delta_rot = np.zeros(3, dtype=np.float32)
        gripper = np.array([0.0], dtype=np.float32)
        action = np.concatenate([delta_pos, delta_rot, gripper])
        return torch.from_numpy(action).unsqueeze(0)


def prepare_observation_batch(obs: Dict[str, Any], device: torch.device, custom_prompt: Optional[str] = None) -> Dict[str, Any]:
    """Chuyển đổi dữ liệu Observation sang dạng PyTorch Batch phù hợp với SmolVLA."""
    batch = {}

    # Extract images and normalize [0, 1] tensor (B, C, H, W)
    images_dict = obs.get("images", {})
    if not images_dict:
        images_dict = {k: v for k, v in obs.items() if k.startswith("image_")}

    for cam_name, img_np in images_dict.items():
        if isinstance(img_np, np.ndarray):
            img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float() / 255.0
            batch[f"observation.images.{cam_name}"] = img_tensor.unsqueeze(0).to(device)

    # Extract joint positions
    if "joint_positions" in obs:
        joint_pos = obs["joint_positions"]
        if isinstance(joint_pos, np.ndarray):
            batch["observation.state"] = torch.from_numpy(joint_pos).float().unsqueeze(0).to(device)

    # Language prompt
    prompt = custom_prompt or obs.get("instruction", "hand over object")
    batch["task"] = [prompt]
    batch["observation.instruction"] = prompt

    return batch


def load_policy(policy_path: Optional[str], dummy: bool, device: torch.device) -> Any:
    """Tải SmolVLAPolicy hoặc Dummy Policy."""
    if dummy or policy_path is None or policy_path == "dummy":
        print("[INFO] Khởi tạo Dummy SmolVLA Policy...")
        return DummySmolVLAPolicy(action_dim=7)

    if not LEROBOT_AVAILABLE:
        print("[LỖI] Thư viện LeRobot chưa được cài đặt trong môi trường hiện tại.")
        sys.exit(1)

    print(f"[INFO] Tải SmolVLAPolicy từ: '{policy_path}'...")
    try:
        policy = SmolVLAPolicy.from_pretrained(policy_path)
        policy.to(device)
        policy.eval()
        print("[INFO] Tải thành công SmolVLAPolicy!")
        return policy
    except Exception as e:
        print(f"[WARN] Không thể tải weights từ '{policy_path}': {e}")
        print("[INFO] Khởi tạo SmolVLAPolicy từ SmolVLAConfig mặc định...")
        config = SmolVLAConfig()
        policy = SmolVLAPolicy(config)
        policy.to(device)
        policy.eval()
        return policy


def run_inference(args):
    config_path = PROJECT_ROOT / "config" / "tasks" / f"{args.task}.yaml"
    if not config_path.exists():
        sys.exit(f"[LỖI] File cấu hình task không tồn tại: {config_path}")

    task_cfg = TaskConfig.from_yaml(str(config_path))
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"[INFO] Device: {device}")

    policy = load_policy(args.policy, args.dummy, device)
    env = ManipulationEnv(task_cfg, gui=args.gui, seed=args.seed)

    obs = env.reset()
    policy.reset()

    prompt = args.prompt or obs.get("instruction", "hand over cube to human")

    print("\n=========================================================")
    print(f" BẮT ĐẦU SUY LUẬN (INFERENCE) SMOLVLA")
    print(f" Task: {args.task} | Prompt: '{prompt}'")
    print(f"=========================================================\n")

    if not args.loop:
        # Single-step inference
        batch = prepare_observation_batch(obs, device, custom_prompt=prompt)
        start_t = time.time()
        with torch.no_grad():
            if hasattr(policy, "select_action"):
                action_out = policy.select_action(batch)
            else:
                action_out = policy(batch)
        elapsed_ms = (time.time() - start_t) * 1000.0

        action_np = action_out.squeeze(0).cpu().numpy() if isinstance(action_out, torch.Tensor) else np.array(action_out)
        
        print(f"Camera inputs: {list(batch.keys())}")
        print(f"Predicted Action (7D): {action_np}")
        print(f"  - Delta Position (x, y, z): {action_np[:3]}")
        print(f"  - Delta Rotation (r, p, y): {action_np[3:6]}")
        print(f"  - Gripper State          : {action_np[6]:.3f}")
        print(f"Inference Latency         : {elapsed_ms:.2f} ms")
        print("=========================================================\n")
        return

    # Multi-step continuous inference loop
    max_steps = args.steps
    print(f"[INFO] Bắt đầu suy luận {max_steps} bước liên tục...")
    
    for step in range(max_steps):
        batch = prepare_observation_batch(obs, device, custom_prompt=prompt)
        start_t = time.time()
        with torch.no_grad():
            if hasattr(policy, "select_action"):
                action_out = policy.select_action(batch)
            else:
                action_out = policy(batch)
        elapsed_ms = (time.time() - start_t) * 1000.0

        action_np = action_out.squeeze(0).cpu().numpy() if isinstance(action_out, torch.Tensor) else np.array(action_out)
        if action_np.ndim > 1:
            action_np = action_np.flatten()
        if len(action_np) < 7:
            action_np = np.pad(action_np, (0, 7 - len(action_np)))

        obs, success, done = env.step(action_np[:7])

        print(f"Step {step+1:03d}/{max_steps:03d} | Latency: {elapsed_ms:.1f}ms | Action pos: ({action_np[0]:.3f}, {action_np[1]:.3f}, {action_np[2]:.3f}) | Success: {success}", flush=True)

        if args.gui:
            time.sleep(1.0 / 30.0)

        if done:
            print(f"[INFO] Episode kết thúc tại bước {step+1}! Success={success}")
            break

    env.close()


def main():
    parser = argparse.ArgumentParser(description="Inference Script độc lập cho SmolVLA Policy")
    parser.add_argument("--task", type=str, default="handover_cube_to_human", help="Tên task config (handover_cube_to_human, pick_cube_to_bowl)")
    parser.add_argument("--policy", type=str, default=None, help="Thư mục checkpoint hoặc HuggingFace repo ID")
    parser.add_argument("--dummy", action="store_true", help="Dùng Dummy Policy thử nghiệm luồng suy luận")
    parser.add_argument("--prompt", type=str, default=None, help="Câu lệnh ngôn ngữ tùy chỉnh (Custom text prompt)")
    parser.add_argument("--device", type=str, default="cuda", help="Thiết bị tính toán ('cuda' hoặc 'cpu')")
    parser.add_argument("--gui", action="store_true", help="Hiển thị màn hình GUI PyBullet")
    parser.add_argument("--loop", action="store_true", help="Chạy vòng lặp suy luận liên tục")
    parser.add_argument("--steps", type=int, default=100, help="Số bước suy luận tối đa khi dùng --loop")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
