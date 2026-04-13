"""力度（Velocity）量化工具。

与 design.md 保持一致：
- 使用“中心对称 μ-law 压扩 + 16 档均匀量化”
- 默认参数：mu=8, center=64, half_range=63, velocity in [1, 127]
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping


@dataclass(frozen=True)
class VelocityConfig:
    """力度分桶映射配置。"""

    num_bins: int = 16
    mapping: str = "mu_law_centered"
    mu: float = 8.0
    center: float = 64.0
    half_range: float = 63.0
    min_velocity: int = 1
    max_velocity: int = 127
    note_off_velocity: int = 0

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "VelocityConfig":
        """从 tokenizer 配置读取力度映射参数。

        兼容两种配置风格：
        - 旧版扁平字段：velocity_bins / velocity_mu / velocity_center / velocity_radius
        - 新版嵌套字段：velocity.num_bins / velocity.mu / velocity.center / velocity.half_range
        """
        velocity_raw = data.get("velocity")
        if velocity_raw is None:
            velocity = {}
        elif isinstance(velocity_raw, Mapping):
            velocity = velocity_raw
        else:
            raise ValueError("`velocity` config must be a mapping when provided.")

        cfg = cls(
            num_bins=int(velocity.get("num_bins", data.get("velocity_bins", cls.num_bins))),
            mapping=str(velocity.get("mapping", cls.mapping)),
            mu=float(velocity.get("mu", data.get("velocity_mu", cls.mu))),
            center=float(velocity.get("center", data.get("velocity_center", cls.center))),
            half_range=float(
                velocity.get("half_range", data.get("velocity_radius", cls.half_range))
            ),
            min_velocity=int(velocity.get("min_velocity", cls.min_velocity)),
            max_velocity=int(velocity.get("max_velocity", cls.max_velocity)),
            note_off_velocity=int(velocity.get("note_off_velocity", cls.note_off_velocity)),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        """校验配置合法性，避免映射过程出现不可预期行为。"""
        if self.mapping != "mu_law_centered":
            raise ValueError(f"Unsupported velocity mapping: {self.mapping}")
        if self.num_bins < 2:
            raise ValueError("num_bins must be >= 2")
        if self.mu <= 0:
            raise ValueError("mu must be > 0")
        if self.half_range <= 0:
            raise ValueError("half_range must be > 0")
        if self.min_velocity >= self.max_velocity:
            raise ValueError("min_velocity must be < max_velocity")


def _clip(value: float, low: float, high: float) -> float:
    """将数值裁剪到给定闭区间 [low, high]。"""
    return min(max(value, low), high)


def velocity_to_bin(velocity: int, cfg: VelocityConfig) -> int:
    """将 MIDI 力度值编码为离散力度桶编号。

    编码流程：
    1. 先将输入力度裁剪到有效范围 [min_velocity, max_velocity]
    2. 归一化到以 center 为中心、half_range 为半幅的区间
    3. 使用 μ-law 压扩，使中间区间更细、两端更稀
    4. 将 [-1,1] 均匀量化到 [0, num_bins-1]
    """
    cfg.validate()

    if velocity == cfg.note_off_velocity:
        velocity = cfg.min_velocity

    v = _clip(float(velocity), float(cfg.min_velocity), float(cfg.max_velocity))
    x = (v - cfg.center) / cfg.half_range
    s = math.copysign(
        math.log1p(cfg.mu * abs(x)) / math.log1p(cfg.mu),
        x,
    )

    k_float = ((s + 1.0) / 2.0) * (cfg.num_bins - 1)
    k = int(round(k_float))
    return int(_clip(float(k), 0.0, float(cfg.num_bins - 1)))


def bin_to_velocity(bin_id: int, cfg: VelocityConfig) -> int:
    """将离散力度桶编号解码为 MIDI 力度代表值。

    解码流程与编码互逆：
    1. 将桶编号映射回压扩域 [-1,1]
    2. 进行 μ-law 反压扩
    3. 还原到 MIDI 力度区间并裁剪
    """
    cfg.validate()
    k = int(_clip(float(bin_id), 0.0, float(cfg.num_bins - 1)))

    s_hat = (2.0 * k / (cfg.num_bins - 1)) - 1.0
    x_hat = math.copysign(((1.0 + cfg.mu) ** abs(s_hat) - 1.0) / cfg.mu, s_hat)
    v_hat = int(round(cfg.center + cfg.half_range * x_hat))
    return int(_clip(float(v_hat), float(cfg.min_velocity), float(cfg.max_velocity)))


def build_velocity_table(cfg: VelocityConfig) -> list[int]:
    """构建每个力度桶的代表力度表（长度等于 num_bins）。"""
    return [bin_to_velocity(i, cfg) for i in range(cfg.num_bins)]
