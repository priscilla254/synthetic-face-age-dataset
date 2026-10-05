"""Stage 3: CodeFormer restoration.

The original run used the upstream CodeFormer command line (fidelity weight
w = 0.7, RetinaFace detection) and no script for it was kept. This wrapper
records the exact invocation so the step is documented and repeatable. The
precise CLI flags used originally were not saved, so they are reconstructed from
the settings reported in the dissertation.
"""
from __future__ import annotations

import subprocess
from pathlib import Path


class FaceRestorer:
    def __init__(self, config):
        self.cfg = config.restoration

    def command(self, input_dir: Path, output_dir: Path) -> list[str]:
        script = Path(self.cfg.codeformer_dir) / "inference_codeformer.py"
        return [
            "python", str(script),
            "-w", str(self.cfg.fidelity_weight),
            "--input_path", str(input_dir),
            "--output_path", str(output_dir),
            "--detection_model", self.cfg.detection_model,
        ]

    def run(self, input_dir: Path, output_dir: Path, dry_run: bool = False) -> list[str]:
        cmd = self.command(Path(input_dir), Path(output_dir))
        if not dry_run:
            subprocess.run(cmd, check=True)
        return cmd
