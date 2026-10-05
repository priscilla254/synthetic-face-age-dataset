from pathlib import Path

from facedata.config import load_config
from facedata.generation.pipeline import describe

ROOT = Path(__file__).resolve().parents[1]


def test_default_config_loads_and_matches_reported_settings():
    cfg = load_config(ROOT / "configs" / "pipeline.yaml")
    assert cfg.seed_generation.num_inference_steps == 50
    assert cfg.seed_generation.guidance_scale == 7.5
    assert cfg.restoration.fidelity_weight == 0.7
    assert cfg.random_seed == 42


def test_pipeline_description_lists_three_stages_and_codeformer_command():
    text = describe(load_config(ROOT / "configs" / "pipeline.yaml"))
    assert "Stable Diffusion XL" in text and "AgeTransGAN" in text and "CodeFormer" in text
    assert "-w 0.7" in text
