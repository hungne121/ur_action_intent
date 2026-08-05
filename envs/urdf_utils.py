"""
urdf_utils.py

Tiện ích xử lý đường dẫn file URDF/XACRO và tự động biên dịch XACRO sang URDF.
"""

import os
import subprocess
from pathlib import Path

# Project root calculation
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Cấu hình danh sách đường dẫn ROS setup.bash cho xacro compile (dễ di động giữa các máy)
ROS_SETUP_PATHS = os.environ.get(
    "ROS_SETUP_PATHS",
    "/opt/ros/humble/setup.bash:/home/hungdao/ur_ws/install/setup.bash"
).split(":")


def _resolve_urdf_path(urdf_path: str) -> str:
    """
    Resolve URDF / XACRO file path dynamically.
    Compiles .xacro to .urdf if needed.
    """
    path_obj = Path(urdf_path)

    # Check absolute or direct relative path first
    if path_obj.exists():
        target_path = path_obj
    elif (_PROJECT_ROOT / path_obj).exists():
        target_path = _PROJECT_ROOT / path_obj
    elif (_PROJECT_ROOT / "urdf" / path_obj.name).exists():
        target_path = _PROJECT_ROOT / "urdf" / path_obj.name
    elif (_PROJECT_ROOT / "urdf" / "objects" / path_obj.name).exists():
        target_path = _PROJECT_ROOT / "urdf" / "objects" / path_obj.name
    else:
        # Pybullet default asset search
        return urdf_path

    # If it's a xacro file or if there's a corresponding .xacro file to compile
    if target_path.suffix == ".xacro":
        urdf_target = target_path.with_suffix("")
        if not urdf_target.name.endswith(".urdf"):
            urdf_target = target_path.with_suffix(".urdf")
        _compile_xacro_if_needed(target_path, urdf_target)
        return str(urdf_target)

    # Check if a .xacro sibling exists for this urdf file
    xacro_sibling = target_path.with_suffix(".urdf.xacro")
    if not xacro_sibling.exists():
        xacro_sibling = target_path.with_name(target_path.stem + ".xacro")
    if xacro_sibling.exists():
        _compile_xacro_if_needed(xacro_sibling, target_path)

    return str(target_path)


def _compile_xacro_if_needed(xacro_path: Path, urdf_path: Path):
    """Compile XACRO file to URDF if outdated or missing using portable ROS paths."""
    if not urdf_path.exists() or xacro_path.stat().st_mtime > urdf_path.stat().st_mtime:
        print(f"[XACRO] Compiling {xacro_path.name} -> {urdf_path.name}...")
        valid_sources = [f"source '{p}' 2>/dev/null" for p in ROS_SETUP_PATHS if Path(p).exists() or "/opt/ros" in p]
        source_cmds = " && ".join(valid_sources) if valid_sources else ":"
        cmd = f"{source_cmds}; xacro '{xacro_path}' > '{urdf_path}'"
        try:
            subprocess.run(cmd, shell=True, executable="/bin/bash", check=True)
            print("[XACRO] Compilation complete!")
        except Exception as e:
            print(f"[XACRO Warning] Failed to compile xacro: {e}")
