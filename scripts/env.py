import pybullet as p
import pybullet_data
import time

# ---- 1. Kết nối ----
physicsClient = p.connect(p.GUI)   # hoặc p.DIRECT nếu không cần hiển thị
p.setGravity(0, 0, -9.81)
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # để load plane.urdf có sẵn

# ---- 2. Load mặt sàn ----
plane_id = p.loadURDF("plane.urdf")

# ---- 3. Load robot arm (đã convert từ xacro) ----
robot_id = p.loadURDF(
    "../urdf/ur3e_susgrip.urdf",
    basePosition=[0, 0, 0],
    useFixedBase=True,   # QUAN TRỌNG: tay robot phải fix base, không thì nó rơi tự do
)

# ---- 4. Load các đồ vật ----
cube_id = p.loadURDF(
    "assets/objects/cube.urdf",
    basePosition=[0.4, 0.0, 0.05],
)
cup_id = p.loadURDF(
    "assets/objects/cup.urdf",
    basePosition=[0.4, 0.2, 0.05],
)

# ---- 5. In ra thông tin joint để biết index từng khớp (rất hay dùng) ----
num_joints = p.getNumJoints(robot_id)
for i in range(num_joints):
    info = p.getJointInfo(robot_id, i)
    print(i, info[1].decode("utf-8"), "type:", info[2])

# ---- 6. Set camera view mặc định (để không phải tự xoay bằng chuột mỗi lần) ----
p.resetDebugVisualizerCamera(
    cameraDistance=1.2,
    cameraYaw=50,
    cameraPitch=-35,
    cameraTargetPosition=[0.3, 0, 0.2],
)

# ---- 7. Vòng lặp simulation ----
while True:
    p.stepSimulation()
    time.sleep(1.0 / 240.0)  # pybullet default timestep 1/240s