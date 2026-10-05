"""Pipeline configuration loaded from YAML (replaces the hard-coded paths)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SeedGenerationConfig:
    model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    height: int = 1024
    width: int = 1024
    prompt_ages: tuple[str, ...] = ("13", "17", "25")
    images_per_gender: int = 20


@dataclass(frozen=True)
class AgingConfig:
    img_size: int = 1024
    n_groups: int = 10
    batch_size: int = 16
    snapshot: str = "./snapshot/140000.pt"


@dataclass(frozen=True)
class RestorationConfig:
    fidelity_weight: float = 0.7  # CodeFormer "w": 0 = quality, 1 = fidelity
    detection_model: str = "retinaface_resnet50"
    codeformer_dir: str = "third_party/CodeFormer"


@dataclass(frozen=True)
class PathsConfig:
    seed_dir: Path = Path("data/images/1_seed")
    aged_dir: Path = Path("data/images/2_aged")
    restored_dir: Path = Path("data/images/3_restored")
    metadata_dir: Path = Path("data/metadata")


@dataclass(frozen=True)
class PipelineConfig:
    seed_generation: SeedGenerationConfig = field(default_factory=SeedGenerationConfig)
    aging: AgingConfig = field(default_factory=AgingConfig)
    restoration: RestorationConfig = field(default_factory=RestorationConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    random_seed: int = 42


def _build(cls, raw: dict[str, Any] | None):
    raw = dict(raw or {})
    for key, value in list(raw.items()):
        if isinstance(value, list):
            raw[key] = tuple(value)
    return cls(**raw)


def load_config(path: str | Path) -> PipelineConfig:
    """Load a YAML file into a :class:`PipelineConfig` (unknown keys raise)."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf8")) or {}
    paths_raw = {k: Path(v) for k, v in (raw.get("paths") or {}).items()}
    return PipelineConfig(
        seed_generation=_build(SeedGenerationConfig, raw.get("seed_generation")),
        aging=_build(AgingConfig, raw.get("aging")),
        restoration=_build(RestorationConfig, raw.get("restoration")),
        paths=PathsConfig(**paths_raw),
        random_seed=int(raw.get("random_seed", 42)),
    )
