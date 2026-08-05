"""
test_human_motion_env.py

Script chạy thử nghiệm môi trường PyBullet với dữ liệu chuyển động người (Text-to-Motion HY-Motion).
Chạy qua 300 khung hình chuyển động "human pick the cup", in thông tin vị trí các khớp và xuất ảnh chụp từ Camera PyBullet.
"""

import os
from pathlib import Path
import pybullet as p
from PIL import Image

from envs.human_motion_env import HumanMotionEnv


def main():
    print("=" * 60)
    print(" BẮT ĐẦU CHẠY THỬ MÔI TRƯỜNG PYBULLET HUMAN MOTION (HY-MOTION)")
    print("=" * 60)

    motion_file = "/home/hungdao/ur_ws/src/ur_action_intent/hy_motion/20260803_101050715_117838a8_000.fbx"
    
    # Khởi tạo môi trường
    env = HumanMotionEnv(
        motion_file=motion_file,
        gui=False,  # Chạy headless để kiểm tra và render ảnh tự động
        human_origin=(0.0, -0.6, 0.0),
        human_scale=0.01,
    )

    total_frames = env.human_player.num_frames
    print(f"-> Đã nạp dữ liệu chuyển động thành công!")
    print(f"-> Tổng số khung hình: {total_frames} frames (Khoảng {env.human_player.times[-1]:.2f}s)")
    
    output_dir = Path("/home/hungdao/ur_ws/src/ur_action_intent/hy_motion/renders")
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_frames = [0, 75, 150, 225, 299]
    saved_images = []

    print("\n-> Bắt đầu tiến hành mô phỏng chuyển động qua các frame...")
    for frame in range(total_frames):
        obs, reward, done, info = env.step()

        if frame in sample_frames:
            r_hand = obs['human_right_hand_pos']
            head = obs['human_head_pos']
            print(f"  [Frame {frame:03d}/{total_frames}] Head pos: {head.round(2)}, Right Hand pos: {r_hand.round(2)}")

            # Render ảnh camera PyBullet
            img_arr = env.render_camera_image(width=640, height=480)
            img_path = output_dir / f"human_motion_frame_{frame:03d}.png"
            Image.fromarray(img_arr).save(img_path)
            saved_images.append(str(img_path))

    print("\n=" * 60)
    print(" HOÀN THÀNH MÔ PHỎNG VÀ LƯU ẢNH CAMERAS!")
    print(f" Đã lưu {len(saved_images)} ảnh render tại: {output_dir}")
    print("=" * 60)

    env.close()


if __name__ == "__main__":
    main()
