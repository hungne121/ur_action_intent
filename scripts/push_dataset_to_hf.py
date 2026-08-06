"""
scripts/push_dataset_to_hf.py

Script đẩy tập dữ liệu LeRobotDataset trong thư mục ./datasets/ lên Hugging Face Hub.

Cách dùng:
    python scripts/push_dataset_to_hf.py --dataset-dir datasets/handover_cube_to_human_20260806_152703 --repo-id username/handover_cube_to_human
"""

import argparse
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from huggingface_hub import HfApi

def main():
    parser = argparse.ArgumentParser(description="Push LeRobot dataset folder to Hugging Face Hub")
    parser.add_argument("--dataset-dir", type=str, required=True, help="Thư mục dataset (VD: datasets/handover_cube_to_human_20260806_152703)")
    parser.add_argument("--repo-id", type=str, required=True, help="Tên repo trên Hugging Face (VD: hungne121/handover_cube_to_human)")
    parser.add_argument("--private", action="store_true", help="Tạo repo ở chế độ Private")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_dir)
    if not dataset_path.exists():
        print(f"[LỖI] Thư mục '{dataset_path}' không tồn tại!")
        sys.exit(1)

    api = HfApi()
    try:
        user_info = api.whoami()
        print(f"[INFO] Đã đăng nhập HuggingFace tài khoản: {user_info['name']}")
    except Exception as e:
        print("[LỖI] Bạn chưa đăng nhập Hugging Face trên máy này!")
        print("Vui lòng chạy lệnh:  hf auth login  (hoặc  huggingface-cli login)  và dán Access Token của bạn vào trước.")
        sys.exit(1)

    print(f"--> Đang upload thư mục '{dataset_path}' lên Hugging Face Hub repo '{args.repo_id}'...")
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)
    api.upload_folder(
        folder_path=str(dataset_path),
        repo_id=args.repo_id,
        repo_type="dataset",
    )
    print(f"[HOÀN THÀNH] Dataset đã được tải lên: https://huggingface.co/datasets/{args.repo_id}")

if __name__ == "__main__":
    main()
