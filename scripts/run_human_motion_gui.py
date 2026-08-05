"""
run_human_motion_gui.py

Chạy môi trường PyBullet 3D GUI hiển thị trực tiếp chuyển động người (Text-to-Motion HY-Motion).
Hỗ trợ phát lặp lại liên tục 30 FPS để người dùng quan sát trực tiếp trên màn hình.
"""

import time
import sys
from pathlib import Path

# Đảm bảo import đúng gói envs
sys.path.append(str(Path(__file__).resolve().parent.parent))

from envs.human_motion_env import HumanMotionEnv


def main():
    print("=" * 65)
    print(" 🚀 ĐANG KHỞI ĐỘNG GIAO DIỆN PYBULLET 3D GUI - HUMAN MOTION (HY-MOTION)")
    print("   Dữ liệu hành động: 'human pick the cup'")
    print("   Bấm Ctrl + C trên Terminal để dừng mô phỏng.")
    print("=" * 65)

    motion_file = "/home/hungdao/ur_ws/src/ur_action_intent/hy_motion/20260803_101050715_117838a8_000.fbx"
    
    # Khởi tạo môi trường PyBullet với GIAO DIỆN GUI (gui=True)
    env = HumanMotionEnv(
        motion_file=motion_file,
        gui=True,
        human_origin=(0.38, -0.65, 0.0),
        human_scale=0.01,
    )

    total_frames = env.human_player.num_frames
    print(f"-> Nạp thành công dữ liệu chuyển động ({total_frames} frames).")
    print("-> Cửa sổ PyBullet 3D đang chạy trực tiếp trên màn hình...")

    loop_count = 0
    try:
        while True:
            obs, reward, done, info = env.step()
            
            if done:
                loop_count += 1
                print(f"   [Vòng lặp {loop_count}] Đã hoàn thành 300 frames. Phát lại từ đầu...")
                time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n-> Người dùng ngắt mô phỏng. Đang đóng cửa sổ PyBullet...")
    finally:
        env.close()
        print("-> Đã đóng PyBullet thành công.")


if __name__ == "__main__":
    main()
