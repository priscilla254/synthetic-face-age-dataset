"""Orchestration: describes the three stages and their on-disk contract."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from facedata.config import PipelineConfig
from facedata.generation.face_restorer import FaceRestorer


@dataclass(frozen=True)
class Stage:
    name: str
    tool: str
    input_dir: Path
    output_dir: Path


def plan(config: PipelineConfig) -> list[Stage]:
    p = config.paths
    return [
        Stage("seed_generation", "Stable Diffusion XL", Path("prompts"), p.seed_dir),
        Stage("age_transformation", "AgeTransGAN", p.seed_dir, p.aged_dir),
        Stage("restoration", "CodeFormer", p.aged_dir, p.restored_dir),
    ]


def describe(config: PipelineConfig) -> str:
    lines = [f"{i}. {s.tool}: {s.input_dir} -> {s.output_dir}" for i, s in enumerate(plan(config), 1)]
    lines.append("   restoration command: " + " ".join(FaceRestorer(config).command(config.paths.aged_dir, config.paths.restored_dir)))
    return "\n".join(lines)
