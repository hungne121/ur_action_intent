"""
generate_bowl.py

Tạo file bowl.urdf dạng "tô/bát" lõm thật sự (không phải khối đặc),
để vật thể khác có thể rơi vào bên trong khi mô phỏng trong pybullet.

Cách làm: ghép nhiều mảnh box nhỏ xoay quanh 1 vòng tròn để tạo thành
tường bowl, cộng thêm 1 đáy phẳng ở giữa. Mỗi mảnh box là 1 <collision>/
<visual> riêng trong CÙNG 1 link -> pybullet coi đây là 1 compound shape,
vẫn hợp lệ vì từng mảnh riêng lẻ là convex.

Chạy: python generate_bowl.py
Kết quả: bowl.urdf trong cùng thư mục.
"""

import math

# ---- Tham số hình dạng bowl, tùy chỉnh tại đây ----
RADIUS = 0.08          # bán kính bowl (m)
WALL_HEIGHT = 0.05     # chiều cao tường bowl (m)
WALL_THICKNESS = 0.006 # độ dày tường (m)
BOTTOM_THICKNESS = 0.006
N_SEGMENTS = 20        # số mảnh box ghép vòng tròn (càng nhiều càng tròn, nhưng nặng hơn)
MASS_TOTAL = 0.15      # khối lượng tổng ước lượng (kg)

def segment_length(radius, n_segments):
    """Độ dài 1 mảnh box xấp xỉ theo dây cung của vòng tròn."""
    angle = 2 * math.pi / n_segments
    return 2 * radius * math.sin(angle / 2) * 1.15  # *1.15 để các mảnh hơi chồng nhau, tránh hở

def build_urdf():
    seg_len = segment_length(RADIUS, N_SEGMENTS)
    mass_per_seg = MASS_TOTAL / (N_SEGMENTS + 1)  # +1 cho đáy

    collisions_visuals = []

    # ---- Đáy bowl (1 cylinder mỏng) ----
    collisions_visuals.append(f"""
    <visual>
      <origin xyz="0 0 {BOTTOM_THICKNESS/2:.4f}" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="{RADIUS:.4f}" length="{BOTTOM_THICKNESS:.4f}"/>
      </geometry>
      <material name="bowl_color">
        <color rgba="0.75 0.75 0.78 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 {BOTTOM_THICKNESS/2:.4f}" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="{RADIUS:.4f}" length="{BOTTOM_THICKNESS:.4f}"/>
      </geometry>
    </collision>""")

    # ---- Các mảnh tường xung quanh ----
    for i in range(N_SEGMENTS):
        angle = 2 * math.pi * i / N_SEGMENTS
        x = RADIUS * math.cos(angle)
        y = RADIUS * math.sin(angle)
        z = WALL_HEIGHT / 2 + BOTTOM_THICKNESS
        yaw = angle + math.pi / 2  # xoay box để mặt dài của nó tiếp tuyến với vòng tròn

        collisions_visuals.append(f"""
    <visual>
      <origin xyz="{x:.4f} {y:.4f} {z:.4f}" rpy="0 0 {yaw:.4f}"/>
      <geometry>
        <box size="{WALL_THICKNESS:.4f} {seg_len:.4f} {WALL_HEIGHT:.4f}"/>
      </geometry>
      <material name="bowl_color">
        <color rgba="0.75 0.75 0.78 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="{x:.4f} {y:.4f} {z:.4f}" rpy="0 0 {yaw:.4f}"/>
      <geometry>
        <box size="{WALL_THICKNESS:.4f} {seg_len:.4f} {WALL_HEIGHT:.4f}"/>
      </geometry>
    </collision>""")

    body = "".join(collisions_visuals)

    urdf = f"""<?xml version="1.0"?>
<!--
  bowl.urdf (auto-generated bởi generate_bowl.py)
  Bowl dạng compound shape: 1 đáy (cylinder) + {N_SEGMENTS} mảnh tường (box)
  ghép vòng tròn, tạo thành hình lõm thật sự -> vật thể có thể rơi vào bên trong
  khi mô phỏng va chạm trong pybullet.

  Kích thước: bán kính {RADIUS}m, cao {WALL_HEIGHT}m.
  Muốn đổi kích thước -> sửa tham số ở đầu generate_bowl.py rồi chạy lại script.
-->
<robot name="bowl">
  <link name="bowl_link">
    <inertial>
      <mass value="{MASS_TOTAL}"/>
      <inertia ixx="0.0003" ixy="0.0" ixz="0.0"
               iyy="0.0003" iyz="0.0"
               izz="0.0004"/>
    </inertial>
{body}
  </link>
</robot>
"""
    return urdf


if __name__ == "__main__":
    urdf_content = build_urdf()
    with open("bowl.urdf", "w") as f:
        f.write(urdf_content)
    print("Đã tạo bowl.urdf")
