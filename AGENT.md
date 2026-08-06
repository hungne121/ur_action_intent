# AGENT README — `ur_action_intent` Package Guide

Tài liệu hướng dẫn dành cho AI Agents và Developers về kiến trúc, cách vận hành, mở rộng task mới và đánh giá model SmolVLA trong môi trường mô phỏng UR3e PyBullet.

---

## 1. Tổng quan Kiến trúc (Architecture Overview)

Gói `ur_action_intent` được thiết kế theo kiến trúc mô-đun hóa, hỗ trợ cả 2 luồng công việc:
1. **Thu thập dữ liệu (Data Collection)** bằng Expert Policy xuất ra chuẩn `LeRobotDataset`.
2. **Đánh giá & Suy luận Policy (SmolVLA Evaluation/Inference)**.

```
                      ┌─────────────────────────────────┐
                      │    config/tasks/<task>.yaml     │
                      └────────────────┬────────────────┘
                                       │ TaskConfig.from_yaml()
                                       ▼
                      ┌─────────────────────────────────┐
                      │       ManipulationEnv           │
                      │  - Robot (UR3e + SusGrip 2F)    │
                      │  - Objects & Spawn Regions      │
                      │  - HumanMotionPlayer            │
                      │  - ScenarioExecutor (HRIBench)  │
                      └────────────────┬────────────────┘
                                       │
                      ┌────────────────┴────────────────┐
                      │ Unified Observation & Action    │
                      └────────┬────────────────┬───────┘
                               │                │
                               ▼                ▼
                     collect_data.py     eval_smolvla.py
                     (Expert Policy)     (SmolVLA / Dummy)
                               │                │
                               ▼                ▼
                       LeRobotDataset      Metrics JSON
```

---

## 2. HRIBench Scenario Script — Lý thuyết & Quy tắc

### 2.1 Kịch bản Tình huống (Scenario Scripts)

Mỗi kịch bản `S` là chuỗi hành động tương tác: `S = {a₁, …, aT}`.  
Mỗi hành động `aₜ` gồm **5 thành phần**:

| Ký hiệu | Tên | Mô tả |
|---|---|---|
| `rₜ` | Active interaction role | Vai trò tương tác đang hoạt động |
| `gₜ` | Semantic goal | Mục tiêu ngữ nghĩa của bước |
| `cₜ` | Temporal/causal constraints | Ràng buộc thời gian / nhân quả |
| `mₜ` | Human motion distribution | Phân phối chuyển động người |
| `ρₜ` | Reward & success conditions | Điều kiện thưởng và tiêu chí thành công |

---

### 2.2 Định nghĩa Vai trò Tương tác (Interaction Roles)

Con người **không phải là bối cảnh tĩnh** — họ được tích hợp trực tiếp vào task qua 3 vai trò:

- **Instructor** (Người hướng dẫn): Robot suy luận hành động dựa trên cử chỉ/chỉ chỏ/thị phạm của người. **Animation thường chạy liên tục** không bị freeze.
- **Collaborator** (Người cộng tác): Robot và người phối hợp chặt chẽ trong không gian gần (VD: object handover, cùng bê vật).
- **Intruder** (Người can thiệp): Con người cố ý gây rối, robot phải detect bất thường và khôi phục quy trình.

> ⚠️ **Quy tắc chọn role_pattern**:
> - Nếu người **chủ động thực hiện hành động** mà robot phải quan sát/phản ứng → dùng `instructor`.
> - Nếu robot và người **phải đồng bộ trao đổi vật** → dùng `collaborator`.
> - Nếu người **cố tình can thiệp** vào quy trình robot → dùng `intruder`.

---

### 2.3 Quy trình Thiết lập Môi trường (Generation Pipeline)

4 giai đoạn để chuyển mô tả hợp tác thành simulation:

1. **Scenario Scripting** — Định nghĩa kịch bản (stages, agents, constraints) trong file YAML `scenario_script:`. Kịch bản đến từ yêu cầu task (do người dùng cung cấp).
2. **Scene Instantiation** — Cấu hình YAML: sắp xếp đồ vật, spawn region, vị trí người, góc camera.
3. **Motion Synthesis** — Dữ liệu chuyển động người từ **HY-Motion-1.0** (FBX/GLTF). Được tinh chỉnh theo ràng buộc bàn tay, hướng thân, hướng tiếp cận để hành vi đo lường được.
4. **Simulation Validation & Filtering** — Tự động lọc episode lỗi: unreachable states, interpenetration quá mức, camera bị che, chuỗi hành động không nhất quán.

---

### 2.4 Quy tắc Viết Scenario YAML (Critical Rules)

> ⚠️ **`motion_end_frame = -1` không có nghĩa là "chạy hết" — nó có nghĩa là "freeze tại frame đó" nếu `motion_end_frame ≥ 0`.**

**Quy tắc quan trọng:**

1. **Để animation chạy liên tục không freeze**: Đặt `motion_end_frame` bằng frame cuối của animation (VD: `146` nếu có 147 frames). KHÔNG đặt `motion_end_frame: 50` nếu muốn animation tiếp tục chạy.

2. **Để freeze người tại 1 frame**: Đặt `motion_start_frame: X` và `motion_end_frame: X` với cùng giá trị X.

3. **`agent: robot`** → `human_frame()` trả về frame cuối (freeze). Không thể vừa có `agent: robot` vừa muốn người tiếp tục di chuyển. Nếu muốn cả 2 cùng hoạt động, dùng `agent: both`.

4. **`max_duration_steps`** phải đủ lớn để animation có thể phát hết trong stage. Ví dụ animation 147 frames @ control_hz=20Hz → cần `max_duration_steps ≥ 147`.

5. **Vị trí người** (`human_motion.origin`):
   - `z = 0.0` + bật `auto_floor_align=True` (mặc định) → chân tự chạm sàn.
   - `y` phải đủ xa để tránh capsule body chui vào bàn: clearance = `|table_y_min| - |person_y_max| - capsule_radius > 0`. Với bàn PyBullet standard (table_y_min = -0.501) và capsule torso r=0.095 → cần `origin_y < -0.596`.
   - Giá trị an toàn đã calibrate: **`origin: [0.38, -1.15, 0.0]`**.

---



## 3. Quy trình Thêm Task Mới (Adding New Tasks)

Để tạo một task mới, **KHÔNG CẦN sửa code Python core** (`base_env.py` hay `scene_config.py`). Chỉ cần tạo 1 file YAML mới trong `config/tasks/<new_task>.yaml`.

### Cấu trúc File YAML Mẫu (HRI Task):

```yaml
task_name: my_hri_task

robot:
  urdf: assets/robot/ur3e_susgrip.urdf
  base_position: [0, 0, 0.62]
  fixed_base: true

human_motion:
  motion_dir: "hy_motion/handover01"
  origin: [0.433, -0.243, 0.340]   # Gốc tọa độ người trong PyBullet
  scale: 0.006                      # Tỷ lệ thu phóng FBX -> PyBullet
  target_joint: "R_Wrist"

objects:
  - name: cube
    urdf: assets/objects/cube.urdf
    spawn_region:
      x: [0.33, 0.42]
      y: [-0.12, 0.12]
      z: [0.65, 0.65]

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

success_condition:
  type: distance_below
  object: cube
  target: human_hand
  threshold: 0.10

episode:
  max_steps: 300
  control_hz: 20
```

---

## 4. Công cụ Calibration Tọa độ Người (`calibrate_human_origin.py`)

Khi thêm dữ liệu chuyển động người mới (file `.fbx` / `.gltf`), khớp tay người có thể nằm ngoài tầm với (reachable workspace) của UR3e.

Dùng script calibration để quét tự động:
```bash
python scripts/calibrate_human_origin.py --motion-dir hy_motion/handover01 --verbose
```
Script sẽ tính toán Forward Kinematics (FK) từ GLTF/FBX animation, quy đổi tọa độ Y-Up (FBX) sang Z-Up (PyBullet), và gợi ý tham số `origin` và `scale` chính xác để `R_Wrist` nằm đúng tầm với robot (`dist ∈ [0.28m, 0.50m]`, `z ∈ [0.70m, 0.95m]`).

---

## 5. Hướng dẫn Chạy Scripts Chính (Execution Guide)

### 5.1 Thu thập Dữ liệu (Data Collection)
Chạy thu thập dữ liệu bằng Expert Policy và xuất ra `LeRobotDataset`:

```bash
# Preview GUI xem trước task (không ghi dữ liệu)
python scripts/collect_data.py --task handover_cube_to_human --preview

# Thu thập 50 episodes thành công (Headless - Tốc độ cao)
python scripts/collect_data.py --task handover_cube_to_human --episodes 50 --no-gui
```

### 5.2 Đánh giá Policy (SmolVLA Evaluation)
Đánh giá tỷ lệ thành công của SmolVLA model qua $N$ episodes:

```bash
# Kiểm thử luồng đánh giá với Dummy Policy
python scripts/eval_smolvla.py --task handover_cube_to_human --dummy --episodes 5 --gui

# Đánh giá checkpoint SmolVLA thực tế
python scripts/eval_smolvla.py --task handover_cube_to_human --policy /path/to/checkpoint --episodes 20
```

---

## 6. Sơ đồ File & Vai trò (File Map)

```
ur_action_intent/
├── config/tasks/               # File cấu hình YAML cho từng task
│   ├── handover_cube_to_human.yaml
│   ├── lift_cube.yaml
│   └── pick_cube_to_bowl.yaml
├── envs/                       # Mô-đun môi trường mô phỏng
│   ├── base_env.py             # ManipulationEnv chính (PyBullet physics, step/reset)
│   ├── scenario.py             # ScenarioScript & ScenarioExecutor (HRIBench logic)
│   ├── scene_config.py         # Dataclasses parse YAML config
│   ├── human_motion_player.py  # Load và playback GLTF/FBX motion
│   ├── gripper.py              # Điều khiển SusGrip 2F gripper & mimic joints
│   └── camera.py               # Render Multi-camera (Eye-in-hand + Static)
├── scripts/                    # Các script chạy chính
│   ├── collect_data.py         # Thu thập LeRobotDataset
│   ├── eval_smolvla.py         # Đánh giá SmolVLA evaluation
│   ├── infer_smolvla.py        # Single-step inference script
│   └── calibrate_human_origin.py # Calibration tọa độ chuyển động người
└── AGENT.md                    # File tài liệu này
```
