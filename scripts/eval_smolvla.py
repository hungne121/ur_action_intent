#!/usr/bin/env python3
"""
eval_smolvla.py

Script đánh giá (Evaluation) và suy luận (Inference) mô hình Vision-Language-Action SmolVLA
trên môi trường mô phỏng UR3e PyBullet (Cube-to-Human Handover, Pick-and-Place, etc.).

Sử dụng:
    1. Đánh giá thử nghiệm (Pipeline check với Dummy/Mock Policy):
        python scripts/eval_smolvla.py --task handover_cube_to_human --dummy --episodes 3 --gui

    2. Đánh giá Checkpoint thực tế (HuggingFace Hub hoặc checkpoint cục bộ):
        python scripts/eval_smolvla.py --task handover_cube_to_human --policy lerobot/smolvla_checkpoints --episodes 10

    3. Chạy đơn bước Inference trực tiếp:
        python scripts/eval_smolvla.py --task handover_cube_to_human --infer-once
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import torch

# Ensure project root is in sys.path
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


# =========================================================
# Mock / Dummy Policy for Pipeline Testing
# =========================================================
class DummySmolVLAPolicy:
    """Mock policy mô phỏng API SmolVLAPolicy khi chưa có checkpoint thật."""
    def __init__(self, action_dim: int = 7):
        self.action_dim = action_dim

    def reset(self):
        pass

    def select_action(self, batch: Dict[str, Any]) -> torch.Tensor:
        # Action giả lập: di chuyển nhẹ theo delta ngẫu nhiên, đóng/mở gripper
        delta_pos = np.random.uniform(-0.01, 0.01, size=3).astype(np.float32)
        delta_rot = np.zeros(3, dtype=np.float32)
        gripper = np.array([0.0], dtype=np.float32)  # fully open or closed
        action = np.concatenate([delta_pos, delta_rot, gripper])
        return torch.from_numpy(action).unsqueeze(0)


# =========================================================
# Observation Preprocessor for SmolVLA
# =========================================================
def prepare_observation_batch(obs: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """
    Biến đổi dữ liệu Observation từ ManipulationEnv sang dạng PyTorch Batch phù hợp với SmolVLA.
    """
    batch = {}

    # Extract images and normalize to [0, 1] tensor (Batch, Channels, Height, Width)
    images_dict = obs.get("images", {})
    if not images_dict:
        # Fallback to individual keys
        images_dict = {k: v for k, v in obs.items() if k.startswith("image_")}

    for cam_name, img_np in images_dict.items():
        if isinstance(img_np, np.ndarray):
            # img_np: (H, W, C) -> (C, H, W) float32 in range [0, 1]
            img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float() / 255.0
            batch[f"observation.images.{cam_name}"] = img_tensor.unsqueeze(0).to(device)

    # Extract robot state
    if "joint_positions" in obs:
        joint_pos = obs["joint_positions"]
        if isinstance(joint_pos, np.ndarray):
            batch["observation.state"] = torch.from_numpy(joint_pos).float().unsqueeze(0).to(device)

    # Extract language instruction
    instruction = obs.get("instruction", "manipulate object")
    batch["task"] = [instruction]
    batch["observation.instruction"] = instruction

    return batch


# =========================================================
# Policy Loader
# =========================================================
def load_policy(policy_path: Optional[str], dummy: bool, device: torch.device) -> Any:
    """Tải SmolVLAPolicy từ checkpoint hoặc sử dụng Dummy Policy."""
    if dummy or policy_path is None or policy_path == "dummy":
        print("[INFO] Khởi tạo Dummy SmolVLA Policy (dùng cho kiểm thử pipeline)...")
        return DummySmolVLAPolicy(action_dim=7)

    if not LEROBOT_AVAILABLE:
        print("[LỖI] Thư viện LeRobot chưa được cài đặt trong môi trường hiện tại.")
        sys.exit(1)

    print(f"[INFO] Tải SmolVLAPolicy từ: {policy_path}...")
    try:
        policy = SmolVLAPolicy.from_pretrained(policy_path)
        policy.to(device)
        policy.eval()
        print("[INFO] Tải thành công SmolVLAPolicy!")
        return policy
    except Exception as e:
        print(f"[LỖI] Không thể tải checkpoint từ '{policy_path}': {e}")
        print("[INFO] Thử khởi tạo mô hình SmolVLA ngẫu nhiên từ SmolVLAConfig...")
        config = SmolVLAConfig()
        policy = SmolVLAPolicy(config)
        policy.to(device)
        policy.eval()
        return policy


# =========================================================
# Inference / Single Episode Rollout
# =========================================================
def evaluate_episode(
    env: ManipulationEnv,
    policy: Any,
    device: torch.device,
    max_steps: int = 300,
    gui_sleep: float = 0.0,
) -> Tuple[bool, int, float]:
    """
    Thực hiện 1 episode mô phỏng với SmolVLA policy.
    Trả về: (is_success, step_count, total_reward)
    """
    obs = env.reset()
    policy.reset()

    step_count = 0
    total_reward = 0.0
    done = False
    success = False

    while not done and step_count < max_steps:
        # Transform observation to batch format
        batch = prepare_observation_batch(obs, device)

        # SmolVLA Inference
        with torch.no_grad():
            if hasattr(policy, "select_action"):
                action_out = policy.select_action(batch)
            else:
                action_out = policy(batch)

        # Convert tensor action to 7D numpy array
        if isinstance(action_out, torch.Tensor):
            action_np = action_out.squeeze(0).cpu().numpy()
        else:
            action_np = np.array(action_out, dtype=np.float32)

        # Ensure action shape is 7D
        if action_np.ndim > 1:
            action_np = action_np.flatten()
        if len(action_np) < 7:
            # Pad to 7D if needed
            action_np = np.pad(action_np, (0, 7 - len(action_np)))
        elif len(action_np) > 7:
            action_np = action_np[:7]

        # Step environment
        obs, reward, done, info = env.apply_action(action_np)

        total_reward += reward
        step_count += 1
        success = info.get("is_success", False)

        if step_count % 50 == 0:
            print(f"  [Progress] Episode Step {step_count}/{max_steps} | Success: {success}", flush=True)

        if gui_sleep > 0:
            time.sleep(gui_sleep)

    return success, step_count, total_reward


# =========================================================
# Main Evaluation Function
# =========================================================
def run_evaluation(args):
    config_path = PROJECT_ROOT / "config" / "tasks" / f"{args.task}.yaml"
    if not config_path.exists():
        sys.exit(f"[LỖI] File cấu hình không tồn tại: {config_path}")

    task_cfg = TaskConfig.from_yaml(str(config_path))
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"[INFO] Thiết bị tính toán (Device): {device}")

    # Load policy
    policy = load_policy(args.policy, args.dummy, device)

    # Initialize environment
    env = ManipulationEnv(task_cfg, gui=args.gui, seed=args.seed)

    # Run multi-episode evaluation
    print(f"\n=========================================================")
    print(f" BẮT ĐẦU ĐÁNH GIÁ SMOLVLA POLICY")
    print(f" Task: {args.task} | Episodes: {args.episodes} | Seed: {args.seed}")
    print(f"=========================================================\n")

    results = []
    successes = 0

    for ep_idx in range(args.episodes):
        ep_seed = args.seed + ep_idx if args.seed is not None else None
        env.rng = np.random.default_rng(ep_seed)

        success, steps, reward = evaluate_episode(
            env=env,
            policy=policy,
            device=device,
            max_steps=task_cfg.episode.max_steps,
            gui_sleep=1.0 / 60.0 if args.gui else 0.0,
        )

        status_str = "SUCCESS" if success else "FAILED"
        if success:
            successes += 1

        print(f"[Episode {ep_idx+1:02d}/{args.episodes:02d}] Status: {status_str:7s} | Steps: {steps:03d} | Total Reward: {reward:.1f}", flush=True)

        results.append({
            "episode": ep_idx + 1,
            "success": success,
            "steps": steps,
            "reward": reward,
            "instruction": env.current_instruction,
        })

    success_rate = (successes / args.episodes) * 100.0
    avg_steps = np.mean([r["steps"] for r in results])

    print(f"\n=========================================================")
    print(f" KẾT QUẢ ĐÁNH GIÁ TỔNG HỢP")
    print(f" Tỷ lệ thành công (Success Rate): {success_rate:.2f}% ({successes}/{args.episodes})")
    print(f" Số bước trung bình (Avg Steps): {avg_steps:.1f}")
    print(f"=========================================================\n")

    # Save output results to JSON
    output_dir = PROJECT_ROOT / "eval_results"
    output_dir.mkdir(exist_ok=True)
    summary_path = output_dir / f"eval_smolvla_{args.task}.json"

    with open(summary_path, "w") as f:
        json.dump({
            "task": args.task,
            "policy": args.policy or ("dummy" if args.dummy else "random"),
            "success_rate": success_rate,
            "episodes": args.episodes,
            "avg_steps": avg_steps,
            "details": results,
        }, f, indent=2)

    print(f"[INFO] Đã lưu kết quả đánh giá tại: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Script đánh giá (Evaluation) chuyên biệt cho SmolVLA Policy trên UR3e PyBullet")
    parser.add_argument("--task", type=str, default="handover_cube_to_human", help="Tên file config task (ví dụ: handover_cube_to_human, pick_cube_to_bowl)")
    parser.add_argument("--policy", type=str, default=None, help="Đường dẫn checkpoint hoặc HuggingFace Repo ID của SmolVLA")
    parser.add_argument("--dummy", action="store_true", help="Sử dụng Dummy SmolVLA Policy để kiểm thử luồng code")
    parser.add_argument("--episodes", type=int, default=5, help="Số lượng episode đánh giá")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Thiết bị tính toán ('cuda' hoặc 'cpu')")
    parser.add_argument("--gui", action="store_true", help="Bật màn hình hiển thị trực quan GUI PyBullet")

    args = parser.parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
