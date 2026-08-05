"""
scene_config.py

Định nghĩa cấu trúc dữ liệu (dataclass) để parse file YAML mô tả 1 task/scene.
Thêm task mới = viết thêm 1 file YAML trong configs/tasks/, KHÔNG cần sửa code này.
"""

from dataclasses import dataclass, field
from typing import Optional
import yaml


@dataclass
class SpawnRegion:
    x: list  # [min, max]
    y: list
    z: list
    randomize_yaw: bool = False

    def sample_position(self, rng):
        x = rng.uniform(self.x[0], self.x[1])
        y = rng.uniform(self.y[0], self.y[1])
        z = rng.uniform(self.z[0], self.z[1])
        yaw = rng.uniform(-3.1416, 3.1416) if self.randomize_yaw else 0.0
        return (x, y, z), yaw


@dataclass
class ObjectSpec:
    name: str
    urdf: str
    spawn_region: SpawnRegion
    fixed_base: bool = False


@dataclass
class RobotSpec:
    urdf: str
    base_position: list = field(default_factory=lambda: [0, 0, 0])
    fixed_base: bool = True
    home_positions: Optional[dict] = None
    gripper_finger_offset: float = 0.080


@dataclass
class CameraSpec:
    name: str
    position: Optional[list] = None
    target: Optional[list] = None
    attach_to_link: Optional[str] = None  # nếu set -> camera gắn trên tay robot (eye-in-hand)
    width: int = 320
    height: int = 240
    fov: float = 60.0


@dataclass
class SuccessCondition:
    type: str          # ví dụ "object_inside", "object_height_above", "distance_below"
    object: str
    target: Optional[str] = None
    threshold: Optional[float] = None


@dataclass
class EpisodeSpec:
    max_steps: int = 300
    control_hz: int = 20


@dataclass
class HumanMotionSpec:
    motion_dir: str
    origin: list = field(default_factory=lambda: [0.38, -0.65, 0.0])
    scale: float = 0.01
    target_joint: str = "R_Wrist"
    joint_radius: float = 0.035


@dataclass
class TaskConfig:
    task_name: str
    robot: RobotSpec
    objects: list  # List[ObjectSpec]
    cameras: list  # List[CameraSpec]
    success_condition: SuccessCondition
    episode: EpisodeSpec
    human_motion: Optional[HumanMotionSpec] = None
    language_instructions: list = field(default_factory=list)  # nhiều câu instruction khác nhau cho cùng task

    @staticmethod
    def from_yaml(path: str) -> "TaskConfig":
        with open(path, "r") as f:
            raw = yaml.safe_load(f)

        robot = RobotSpec(**raw["robot"])

        objects = []
        for obj in raw["objects"]:
            region = SpawnRegion(**obj.pop("spawn_region"))
            objects.append(ObjectSpec(spawn_region=region, **obj))

        cameras = [CameraSpec(**c) for c in raw.get("cameras", [])]
        success = SuccessCondition(**raw["success_condition"])
        episode = EpisodeSpec(**raw.get("episode", {}))
        human_motion = HumanMotionSpec(**raw["human_motion"]) if "human_motion" in raw else None

        return TaskConfig(
            task_name=raw["task_name"],
            robot=robot,
            objects=objects,
            cameras=cameras,
            success_condition=success,
            episode=episode,
            human_motion=human_motion,
            language_instructions=raw.get("language_instructions", []),
        )
