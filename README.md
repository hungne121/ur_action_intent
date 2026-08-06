# ur_action_intent

<<<<<<< HEAD
Mô phỏng robot arm UR3e tích hợp tay kẹp SusGrip 2F trong PyBullet, phục vụ huấn luyện và đánh giá chính sách hành động (action policy) SmolVLA / VLA.
=======
Mô phỏng robot arm UR3e / UR5 tích hợp tay kẹp SusGrip 2F / Robotiq 85 trong PyBullet, phục vụ thu thập dữ liệu `LeRobotDataset`, huấn luyện và đánh giá chính sách hành động (action policy) SmolVLA / VLA cho các tác vụ thao tác tương tác Người - Robot (HRI).
>>>>>>> 13f9d2b (feat: implement SmolVLA policy components and add ScenarioExecutor for environment stage management)

---

## 1. Môi trường hoạt động (Conda Environment)

Tất cả các lệnh chạy trong dự án cần được thực thi trong môi trường Conda **`ur_bullet312`**:

```bash
# Kích hoạt môi trường conda
conda activate ur_bullet312
```

---

## 2. Hướng dẫn Lệnh Chạy (Commands Quickstart)

### 2.1 Thu thập dữ liệu (`scripts/collect_data.py`)
Script thu thập dữ liệu tự động cho các task được cấu hình trong `config/tasks/` (bao gồm cả các task HRI Scenario Script).

```bash
# Preview GUI xem trước task (không ghi dữ liệu)
python scripts/collect_data.py --task handover_cube_to_human --preview

# Chạy thu thập 50 episode ẩn (HEADLESS - Nhanh hơn) cho task 'handover_cube_to_human'
python scripts/collect_data.py --task handover_cube_to_human --episodes 50 --no-gui
```

### 2.2 Đánh giá SmolVLA Policy (`scripts/eval_smolvla.py`)
Script đánh giá tỷ lệ thành công của SmolVLA policy qua $N$ episodes:

```bash
<<<<<<< HEAD
lerobot-train \
  --dataset.repo_id=local/lift_cube_20260805_162926 \
  --dataset.root=./datasets/lift_cube_20260805_162926 \
=======
# Kiểm thử luồng đánh giá với Dummy Policy
python scripts/eval_smolvla.py --task handover_cube_to_human --dummy --episodes 5 --gui

# Đánh giá checkpoint SmolVLA thực tế
python scripts/eval_smolvla.py --task handover_cube_to_human --policy /path/to/checkpoint --episodes 20
```

### 2.3 Calibrate Tọa độ Người (`scripts/calibrate_human_origin.py`)
Tự động quét và tìm vị trí `origin` / `scale` chuẩn cho motion data người (FBX/GLTF) vừa tầm với robot UR3e:

```bash
python scripts/calibrate_human_origin.py --motion-dir hy_motion/handover01 --verbose
```

### 2.4 Huấn luyện LeRobot SmolVLA (`lerobot-train`)
```bash
lerobot-train \
  --dataset.repo_id=local/handover_cube_to_human_20260805_175029 \
  --dataset.root=./datasets/handover_cube_to_human_20260805_175029 \
>>>>>>> 13f9d2b (feat: implement SmolVLA policy components and add ScenarioExecutor for environment stage management)
  --policy.path=lerobot/smolvla_base \
  --output_dir=outputs/train/smolvla_ur3_handover \
  --job_name=smolvla_ur3_handover \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --steps=10000 \
  --batch_size=32
```

---

## ⚙️ 3. Cấu hình Task HRIBench Scenario Script (`config/tasks/*.yaml`)

<<<<<<< HEAD
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

## ⚙️ 3. Cấu hình Task HRIBench Scenario Script (`config/tasks/*.yaml`)

Môi trường `ManipulationEnv` trong `envs/base_env.py` hoàn toàn được điều khiển bởi file cấu hình YAML. Đối với các tác vụ HRI, hệ thống sử dụng abstraction **`scenario_script`** (triết lý HRIBench):

```yaml
task_name: handover_cube_to_human

# Cấu hình Robot UR3e
robot:
  urdf: assets/robot/ur3e_susgrip.urdf
  base_position: [0, 0, 0.62]
  fixed_base: true

# Cấu hình đĩa chuyển động người (FBX/GLTF) đã calibrate
human_motion:
  motion_dir: "hy_motion/handover01"
  origin: [0.433, -0.243, 0.340]
  scale: 0.006
  target_joint: "R_Wrist"

# Danh sách vật thể
objects:
  - name: cube
    urdf: assets/objects/cube.urdf
    spawn_region:
      x: [0.33, 0.42]
      y: [-0.12, 0.12]
      z: [0.65, 0.65]

# Kịch bản tương tác HRIBench (Collaborator Role)
scenario_script:
  role_pattern: collaborator
  stages:
    - id: signal
      agent: human
      motion_start_frame: 0
      motion_end_frame: 50
      max_duration_steps: 50
    - id: pick
      agent: robot
      expert_phases: [approach, pre_descend, descend, close, hold_close]
    - id: handover
      agent: both
      expert_phases: [lift, move_to_target, place]
      motion_start_frame: 50
      motion_end_frame: 50

# Camera Multi-view (Eye-in-hand + 2 góc cố định)
cameras:
  - name: camera1
    position: [1.35, -0.30, 1.15]
    target: [0.25, -0.30, 0.75]
    fov: 65
  - name: camera2
    position: [0.60, 0.00, 1.45]
    target: [0.38, 0.00, 0.78]
    fov: 50
  - name: camera3
    attach_to_link: "tool0"
    fov: 60

success_condition:
  type: distance_below
  object: cube
  target: human_hand
  threshold: 0.10

episode:
  max_steps: 300
  control_hz: 20

language_instructions:
  - "hand over the cube to the human"
  - "pick up the cube and give it to the person"
  - "pass the block to the person's hand"
```

---

## 4. Cấu trúc Thư mục Dự án

```text
ur_action_intent/
├── AGENT.md                    # Tài liệu kỹ thuật chi tiết dành cho Agent / Developers
├── README.md                   # Hướng dẫn sử dụng nhanh
├── config/
│   └── tasks/                  # Các file cấu hình Task (handover_cube_to_human.yaml, lift_cube.yaml,...)
├── envs/                       # Mô phỏng PyBullet Core
│   ├── base_env.py             # Class môi trường chính ManipulationEnv
│   ├── scenario.py             # ScenarioScript & ScenarioExecutor (HRIBench logic)
│   ├── scene_config.py         # Parse YAML config & Dataclasses
│   ├── human_motion_player.py  # FBX/GLTF C-library loader & joint trajectory player
│   ├── gripper.py              # Controller kẹp SusGrip 2F & mimic joints
│   └── camera.py               # Render Multi-camera system
└── scripts/
    ├── collect_data.py         # Thu thập LeRobotDataset từ Expert Policy
    ├── eval_smolvla.py         # Script đánh giá (Evaluation) SmolVLA Policy
    ├── infer_smolvla.py        # Single-step inference script
    └── calibrate_human_origin.py # Calibration tọa độ người trong workspace UR3e
```
