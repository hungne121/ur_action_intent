"""
data_collect.py

Collect SmolVLA training data using the UR5 pick-cube environment.
  - Camera 1 : front-side view  (kept from ur5_smolvla_env)
  - Camera 2 : overhead view    (kept from ur5_smolvla_env)
  - Camera 3 : eye-in-hand camera mounted on the robot end-effector  ← REPLACED

Usage:
    # Preview (GUI, no data saved)
    python scripts/data_collect.py --preview

    # Collect 100 episodes headlessly
    python scripts/data_collect.py --episodes 100 --no-gui

    # Collect 200 episodes with GUI
    python scripts/data_collect.py --episodes 200 --gui
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pybullet as p

# ── Make sure the project root is on sys.path so we can import ur5_smolvla_env ──
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from ur5_smolvla_env import (  # noqa: E402
    GUI_WINDOW_SIZE,
    OBSERVATION_CAMERA_SPECS,
    UR5PickCubeEnv,
    build_lerobot_dataset,
    make_expert_action,
    phase_done,
)


# =========================================================
# Eye-in-hand camera helper
# =========================================================
def render_eye_in_hand(robot_id, eef_link_id, image_size, fov=60, gui=False):
    """
    Render an RGB image from a camera mounted on the robot end-effector.

    The camera frame is the ee_link frame (eef_link_id = ee_fixed_joint child).
    Axes used (tuned for ur5_robotiq_85.urdf):
        rot_mat[:, 0]  →  local X  →  camera forward / looking direction
        rot_mat[:, 2]  →  local Z  →  camera up vector

    Parameters
    ----------
    robot_id     : PyBullet body ID of the robot
    eef_link_id  : link index of the ee_link (7 for ur5_robotiq_85.urdf)
    image_size   : (width, height) tuple
    fov          : camera field of view in degrees
    gui          : use OpenGL renderer when True, tiny renderer otherwise
    """
    width, height = image_size

    # Get ee_link world pose (index [4]=pos, [5]=orn, computeForwardKinematics=True)
    ee_state = p.getLinkState(robot_id, eef_link_id, computeForwardKinematics=True)
    ee_pos = np.array(ee_state[4])   # world position
    ee_orn = ee_state[5]             # world orientation (quaternion)

    # 3×3 rotation matrix — each column is a local axis in world frame
    rot_mat = np.array(p.getMatrixFromQuaternion(ee_orn)).reshape(3, 3)
    forward = rot_mat[:, 2]   # local X  → looking direction
    up      = rot_mat[:, 2]   # local Z  → camera up (pose-dependent)

    # Offset cam_eye slightly to avoid clipping into the gripper mesh
    cam_eye    = ee_pos + forward * 0.04 + rot_mat[:, 1] * 0.1
    cam_target = ee_pos + forward * 0.5

    view_matrix = p.computeViewMatrix(
        cameraEyePosition=cam_eye.tolist(),
        cameraTargetPosition=cam_target.tolist(),
        cameraUpVector=up.tolist(),
    )
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=fov,
        aspect=float(width) / height,
        nearVal=0.01,
        farVal=3.0,
    )

    _, _, rgb, _, _ = p.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL if gui else p.ER_TINY_RENDERER,
    )
    rgb = np.reshape(rgb, (height, width, 4))[:, :, :3].astype(np.uint8)
    return rgb


# =========================================================
# Env subclass — only camera 3 is overridden
# =========================================================
class UR5PickCubeEnvEyeInHand(UR5PickCubeEnv):
    """
    Identical to UR5PickCubeEnv except:
        camera 3  →  eye-in-hand camera on the robot end-effector.
    """

    # ------------------------------------------------------------------
    # Override only the camera-3 methods
    # ------------------------------------------------------------------
    def get_camera3_image(self):
        """Return eye-in-hand RGB image (replaces fixed side camera)."""
        return render_eye_in_hand(
            robot_id=self.robot.id,
            eef_link_id=self.robot.eef_id,   # 7 = ee_fixed_joint child (ee_link)
            image_size=self.image_size,
            fov=60,
            gui=self.gui,
        )

    def capture_observation_images(self):
        """
        Camera 1 : front-side  (fixed, from OBSERVATION_CAMERA_SPECS[0])
        Camera 2 : overhead    (fixed, from OBSERVATION_CAMERA_SPECS[1])
        Camera 3 : eye-in-hand (dynamic, computed from EE pose each step)
        """
        return {
            "image_camera1": self.render_camera(**OBSERVATION_CAMERA_SPECS[0]),
            "image_camera2": self.render_camera(**OBSERVATION_CAMERA_SPECS[1]),
            "image_camera3": self.get_camera3_image(),
        }


# =========================================================
# Data collection
# =========================================================
def collect_dataset(
    repo_id: str = "local/ur5_pick_cube_eye_in_hand",
    root: str = "./lerobot_dataset_eye_in_hand",
    num_episodes: int = 100,
    gui: bool = False,
    seed: int = 42,
):
    """Collect successful expert demonstrations into a LeRobot dataset."""
    root_path = Path(root)
    if root_path.exists():
        print(f"[INFO] Removing existing dataset folder: {root_path.resolve()}")
        shutil.rmtree(root_path)

    env = UR5PickCubeEnvEyeInHand(
        gui=gui,
        image_size=(224, 224),
        seed=seed,
        observation_camera_count=3,
    )
    dataset = build_lerobot_dataset(
        repo_id=repo_id,
        root=root,
        image_size=(224, 224),
        fps=10,
    )

    saved_count = 0
    dropped_count = 0

    try:
        for ep_idx in range(num_episodes):
            obs = env.reset()

            phases = [
                "approach_cube",
                "pre_descend",
                "descend",
                "close",
                "hold_close",
                "lift",
            ]
            phase_idx = 0
            phase_step_count = 0
            success = False

            for _ in range(320):
                phase = phases[phase_idx]
                action = make_expert_action(env, phase)

                # Randomly pick Chinese or English task description
                task = obs["task_zh"] if env.rng.random() < 0.5 else obs["task_en"]

                # Record frame: camera1, camera2, eye-in-hand (camera3), state, action
                frame = {
                    "observation.images.camera1": obs["image_camera1"],
                    "observation.images.camera2": obs["image_camera2"],
                    "observation.images.camera3": obs["image_camera3"],  # eye-in-hand
                    "observation.state": obs["joint_positions"],
                    "action": action.astype(np.float32),
                    "task": task,
                }
                dataset.add_frame(frame)

                obs, _reward, done, _ = env.apply_action(action)
                phase_step_count += 1

                if phase_done(env, phase, phase_step_count):
                    phase_idx += 1
                    phase_step_count = 0
                    if phase_idx >= len(phases):
                        done = True

                if done:
                    success = env.is_success()
                    break

            if success:
                dataset.save_episode()
                saved_count += 1
                print(
                    f"[Episode {ep_idx:03d}]  SAVED  "
                    f"task='{env.language_instruction_en}'  "
                    f"cube={np.round(env.cube_spawn_pos, 3).tolist()}  "
                    f"(saved={saved_count}, dropped={dropped_count})"
                )
            else:
                dataset.clear_episode_buffer(delete_images=True)
                dropped_count += 1
                print(
                    f"[Episode {ep_idx:03d}]  DROPPED  "
                    f"task='{env.language_instruction_en}'  "
                    f"cube={np.round(env.cube_spawn_pos, 3).tolist()}  "
                    f"(saved={saved_count}, dropped={dropped_count})"
                )

    finally:
        dataset.finalize()
        env.disconnect()
        print("\n=========================================================")
        print(f"[FINAL]  saved={saved_count}  dropped={dropped_count}")
        print("=========================================================")


# =========================================================
# Preview (GUI, no data saved)
# =========================================================
def preview_environment(seed: int = 42, seconds: float = 20.0):
    """Open GUI preview with all three cameras (including eye-in-hand)."""
    env = UR5PickCubeEnvEyeInHand(
        gui=True,
        image_size=(224, 224),
        seed=seed,
        observation_camera_count=3,
    )
    try:
        env.reset()
        for _ in range(int(seconds * 240)):
            env.step_sim(1)
    finally:
        env.disconnect()


# =========================================================
# CLI
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect SmolVLA training data — camera 3 = eye-in-hand"
    )
    parser.add_argument("--preview", action="store_true",
                        help="Open GUI preview only (no data saved)")
    parser.add_argument("--preview-seconds", type=float, default=20.0)
    parser.add_argument("--episodes", type=int, default=100,
                        help="Number of episodes to collect")
    parser.add_argument("--repo-id", default="local/ur5_pick_cube_eye_in_hand")
    parser.add_argument("--root", default="./lerobot_dataset_eye_in_hand",
                        help="Local path to save the dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-gui", action="store_true",
                        help="Run in DIRECT mode (faster, no window)")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.preview:
        preview_environment(seed=args.seed, seconds=args.preview_seconds)
        return

    collect_dataset(
        repo_id=args.repo_id,
        root=args.root,
        num_episodes=args.episodes,
        gui=not args.no_gui,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
