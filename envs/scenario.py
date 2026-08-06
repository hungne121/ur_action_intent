"""
scenario.py
-----------
Hai thành phần cho HRI task có cấu trúc stage-based:

  ScenarioScript  : Config (đọc từ YAML) mô tả kịch bản — agents, stages,
                    temporal dependencies, role pattern.

  ScenarioExecutor: Runtime state machine điều phối staging và timing human
                    motion. Importable độc lập, không phụ thuộc ManipulationEnv.

Thêm task HRI mới: Chỉ cần thêm khối 'scenario_script:' vào YAML.
Task cũ (không có khối đó): Hoạt động như cũ, không ảnh hưởng.

Ví dụ YAML:
    scenario_script:
      role_pattern: collaborator   # instructor | collaborator | intruder
      stages:
        - id: signal
          agent: human
          expert_phases: []
          motion_start_frame: 0
          motion_end_frame: 50
          max_duration_steps: 50
        - id: pick
          agent: robot
          expert_phases: [approach, pre_descend, descend, close, hold_close]
          max_duration_steps: 80
        - id: handover
          agent: both
          expert_phases: [lift, move_to_target, place]
          motion_start_frame: 50
          motion_end_frame: 50
          max_duration_steps: 150
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ══════════════════════════════════════════════════════════════
# Config — đọc từ YAML
# ══════════════════════════════════════════════════════════════

@dataclass
class ScenarioStage:
    """Một giai đoạn trong kịch bản tương tác."""
    id: str                                         # "signal" | "pick" | "lift" | "handover"
    agent: str                                      # "human" | "robot" | "both"
    description: str = ""
    expert_phases: List[str] = field(default_factory=list)  # phases cho expert policy
    motion_start_frame: int = 0                     # frame bắt đầu cho HumanMotionPlayer
    motion_end_frame: int = -1                      # -1 = freeze ở frame này
    max_duration_steps: int = 9999                  # timeout tự động chuyển stage


@dataclass
class ScenarioScript:
    """Cấu hình kịch bản đầy đủ — đọc từ YAML."""
    role_pattern: str = "collaborator"              # "instructor" | "collaborator" | "intruder"
    stages: List[ScenarioStage] = field(default_factory=list)

    @staticmethod
    def from_dict(d: dict) -> "ScenarioScript":
        """Parse từ dict đọc bởi yaml.safe_load."""
        stages = [ScenarioStage(**s) for s in d.get("stages", [])]
        return ScenarioScript(
            role_pattern=d.get("role_pattern", "collaborator"),
            stages=stages,
        )


# ══════════════════════════════════════════════════════════════
# Runtime — state machine điều phối execution
# ══════════════════════════════════════════════════════════════

class ScenarioExecutor:
    """
    Điều phối stage transitions và timing human motion theo ScenarioScript.

    Không phụ thuộc ManipulationEnv — có thể import và sử dụng độc lập:

        executor = ScenarioExecutor(script)
        executor.reset()

        while not done:
            frame = executor.human_frame()    # → HumanMotionPlayer.update(frame)
            ctx   = executor.obs_context()   # → obs.update(ctx)
            executor.tick()

            if phase_done(current_phase):
                executor.advance_phase()
    """

    def __init__(self, script: ScenarioScript):
        self.script = script
        self.reset()

    def reset(self):
        self._stage_idx = 0
        self._stage_step = 0         # bước trong stage hiện tại
        self._human_frame = 0        # frame cuối đã phát (dùng khi freeze)
        self._phase_idx = 0          # index trong expert_phases của stage hiện tại

    # ── Stage access ──────────────────────────────────────────────

    @property
    def current_stage(self) -> Optional[ScenarioStage]:
        if self._stage_idx < len(self.script.stages):
            return self.script.stages[self._stage_idx]
        return None

    @property
    def stage_id(self) -> str:
        s = self.current_stage
        return s.id if s else "done"

    def is_last_stage(self) -> bool:
        """Trả về True nếu executor đang ở stage cuối cùng của script."""
        return self._stage_idx >= len(self.script.stages) - 1

    def is_complete(self) -> bool:
        """Trả về True nếu đã hoàn thành tất cả các stages."""
        return self._stage_idx >= len(self.script.stages)

    @property
    def current_phase(self) -> str:
        """Phase hiện tại cho expert policy."""
        s = self.current_stage
        if not s or not s.expert_phases:
            return "hold"
        idx = min(self._phase_idx, len(s.expert_phases) - 1)
        return s.expert_phases[idx]

    @property
    def expert_phases(self) -> List[str]:
        """Expert phases của stage hiện tại (alias cho current_stage.expert_phases)."""
        s = self.current_stage
        return s.expert_phases if s else []

    @property
    def all_robot_phases(self) -> List[str]:
        """Toàn bộ expert phases của các robot-stages, theo thứ tự."""
        phases: List[str] = []
        for stage in self.script.stages:
            if stage.agent in ("robot", "both"):
                phases.extend(stage.expert_phases)
        return phases

    # ── Lifecycle ─────────────────────────────────────────────────

    def tick(self):
        """Gọi mỗi simulation step để tăng bộ đếm."""
        self._stage_step += 1

    def advance_stage(self):
        """Chuyển sang stage tiếp theo."""
        self._stage_idx += 1
        self._stage_step = 0
        self._phase_idx = 0

    def advance_phase(self):
        """Chuyển sang phase tiếp theo trong stage hiện tại.
        Tự động advance stage nếu đã hết phases."""
        s = self.current_stage
        if s and self._phase_idx < len(s.expert_phases) - 1:
            self._phase_idx += 1
        else:
            self.advance_stage()

    def is_complete(self) -> bool:
        return self._stage_idx >= len(self.script.stages)

    def is_stage_timeout(self) -> bool:
        s = self.current_stage
        return s is not None and self._stage_step >= s.max_duration_steps

    # ── Human motion ──────────────────────────────────────────────

    def human_frame(self) -> int:
        """Frame index để truyền vào HumanMotionPlayer.update()."""
        s = self.current_stage
        if s is None or s.agent not in ("human", "both"):
            return self._human_frame      # freeze ở frame cuối
        frame = s.motion_start_frame + self._stage_step
        if s.motion_end_frame >= 0:
            frame = min(frame, s.motion_end_frame)
        self._human_frame = frame
        return frame

    # ── Observation context ───────────────────────────────────────

    def obs_context(self) -> Dict[str, object]:
        """Dict thêm vào obs của env — SmolVLA và expert policy đều thấy."""
        s = self.current_stage
        return {
            "scenario_stage":      s.id if s else "done",
            "human_signal_active": s.agent in ("human", "both") if s else False,
        }
