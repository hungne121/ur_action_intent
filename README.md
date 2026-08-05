# ur_action_intent

Mô phỏng robot arm UR3e tích hợp tay kẹp SusGrip 2F trong PyBullet, phục vụ huấn luyện và đánh giá chính sách hành động (action policy) SmolVLA / VLA.

---

## 🛠️ 1. Môi trường hoạt động (Conda Environment)

Tất cả các lệnh chạy trong dự án cần được thực thi trong môi trường Conda **`ur_bullet`**:

```bash
# Kích hoạt môi trường conda
conda activate ur_bullet
```

---

## 🚀 2. Hướng dẫn Lệnh Chạy (Commands Quickstart)

### 2.1 Thu thập dữ liệu theo Task Config YAML (`scripts/collect_data.py`)
Script thu thập dữ liệu tự động cho các task được cấu hình trong `config/tasks/`.

```bash
# Chạy thu thập 20 episode có giao diện GUI cho task 'pick_cube_to_bowl'
python scripts/collect_data.py --task pick_cube_to_bowl --episodes 20 --gui

# Chạy thu thập 50 episode ẩn (HEADLESS - Nhanh hơn) cho task 'lift_cube'
python scripts/collect_data.py --task lift_cube --episodes 50 --no-gui
```

**Các tham số (CLI Arguments):**
| Tham số | Kiểu dữ liệu | Mặc định | Mô tả |
| :--- | :--- | :--- | :--- |
| `--task` | `str` | *(Bắt buộc)* | Tên file task YAML trong `config/tasks/` (không cần đuôi `.yaml`). Ví dụ: `pick_cube_to_bowl`, `lift_cube`. |
| `--episodes` | `int` | `10` | Số lượng episode cần chạy thu thập. |
| `--gui` / `--no-gui` | `flag` | `--gui` | `--gui`: Bật cửa sổ mô phỏng PyBullet.<br>`--no-gui`: Chạy ẩn ở chế độ DIRECT (tốc độ cao). |
| `--seed` | `int` | `0` | Seed ngẫu nhiên khởi tạo vị trí vật thể và chọn câu lệnh ngôn ngữ. |

---

### 2.2 Training Lerobot

```bash
lerobot-train \
  --dataset.repo_id=local/lift_cube_20260805_162926 \
  --dataset.root=./datasets/lift_cube_20260805_162926 \
  --policy.path=lerobot/smolvla_base \
  --output_dir=outputs1/train/smolvla_ur3_liftcube \
  --job_name=smolvla_ur5_pick_cube \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --steps=10000 \
  --batch_size=32

```

---

### 2.3 Demo Camera Eye-In-Hand độc lập (`scripts/eye_in_hand.py`)
Chạy mô phỏng kiểm tra tính toán ma trận góc nhìn Eye-In-Hand camera 3D động bám theo End-Effector:

```bash
python scripts/eye_in_hand.py
```

---

### 2.4 Tạo mô hình URDF chiếc bát (`scripts/generate_bowl.py`)
Tạo tự động file URDF chiếc bát (`urdf/objects/bowl.urdf`):

```bash
python scripts/generate_bowl.py
```

---

## ⚙️ 3. Cấu hình Task qua File YAML (`config/tasks/*.yaml`)

Môi trường `ManipulationEnv` trong `scripts/base_env.py` hoàn toàn được điều khiển bởi file cấu hình YAML. Để tạo task mới, chỉ cần thêm 1 file YAML vào `config/tasks/`.

### 📋 Cấu trúc một file Task Config (`pick_cube_to_bowl.yaml`)

```yaml
task_name: pick_cube_to_bowl

# Cấu hình Robot
robot:
  urdf: assets/robot/ur3e_susgrip.urdf
  base_position: [0, 0, 0]
  fixed_base: true

# Danh sách vật thể xuất hiện trong mô phỏng
objects:
  - name: cube
    urdf: assets/objects/cube.urdf
    fixed_base: false
    spawn_region:
      x: [0.30, 0.45]
      y: [-0.15, 0.15]
      z: [0.05, 0.05]
      randomize_yaw: true

  - name: bowl
    urdf: assets/objects/bowl.urdf
    fixed_base: false
    spawn_region:
      x: [0.35, 0.40]
      y: [0.15, 0.25]
      z: [0.01, 0.01]
      randomize_yaw: false

# Cấu hình Camera quan sát (Chuẩn 3 camera cho SmolVLA: camera1, camera2, camera3)
cameras:
  - name: camera1
    position: [0.90, -0.55, 1.05]
    target: [0.38, 0.00, 0.78]
    fov: 55
    width: 224
    height: 224
  - name: camera2
    position: [0.60, 0.00, 1.45]
    target: [0.38, 0.00, 0.78]
    fov: 50
    width: 224
    height: 224
  - name: camera3
    attach_to_link: "tool0"   # Camera Eye-In-Hand động gắn tại tool0
    fov: 60
    width: 224
    height: 224


# Điều kiện hoàn thành Task (Success Condition)
success_condition:
  type: object_inside   # Các loại: 'object_inside', 'object_height_above', 'distance_below'
  object: cube
  target: bowl
  threshold: 0.08

# Giới hạn episode
episode:
  max_steps: 300
  control_hz: 20

# Danh sách các câu lệnh ngôn ngữ đa dạng cho bài toán
language_instructions:
  - "pick up the cube and place it in the bowl"
  - "put the block into the bowl"
  - "move the cube to the bowl"
```

---

## 📁 4. Cấu trúc Thư mục Dự án

```text
ur_action_intent/
├── config/
│   └── tasks/                  # Nơi chứa các file cấu hình Task YAML (lift_cube.yaml, pick_cube_to_bowl.yaml,...)
├── envs/                       # Python module export ManipulationEnv & TaskConfig
├── scripts/
│   ├── base_env.py             # Class môi trường chính ManipulationEnv (PyBullet simulation, IK, Mimic Joints, Eye-In-Hand)
│   ├── scene_config.py         # Dataclass parse file YAML
│   ├── collect_data.py         # Script thu thập dữ liệu tổng quát theo Task YAML
│   ├── data_collect.py         # Script thu thập LeRobotDataset cho SmolVLA
│   ├── eye_in_hand.py          # Demo tính toán ma trận Eye-In-Hand camera 3D
│   └── generate_bowl.py        # Tạo file URDF cái bát
├── urdf/                       # Các file robot URDF/XACRO & vật thể mô phỏng
└── README.md
```
