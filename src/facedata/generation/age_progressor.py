"""Stage 2: age progression / regression with AgeTransGAN.

Restructured from ``legacy_as_run/generation/2.age_photographs.py``; not re-run.
AgeTransGAN (github.com/AvLab-CV/AgeTransGAN) targets Python 3.6 / PyTorch 1.4 and
must live in its own environment; its ``lm_process``, ``img_process`` and
``age_transfer`` modules are imported from that checkout (``--agetransgan-dir``).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from facedata.dataset.age_bins import AGE_TAG_TO_GROUP


def age_tag(index: int) -> str:
    """0-based output position -> tag used in filenames ('age1' .. )."""
    return f"age{index + 1}"


def aged_filename(base_stem: str, index: int) -> str:
    """'img025' + position 2 -> 'img025_age3.png' (base id kept in the name)."""
    return f"{base_stem}_{age_tag(index)}.png"


def group_for_tag(tag: str) -> str | None:
    return AGE_TAG_TO_GROUP.get(tag)


@dataclass
class AgedImage:
    base_filename: str
    aged_filename: str
    age_tag: str


class AgeProgressor:
    def __init__(self, config, agetransgan_dir: Path):
        self.cfg = config.aging
        self.repo = Path(agetransgan_dir)
        self._model = None

    def _load(self):
        if self._model is None:
            sys.path.insert(0, str(self.repo / "test"))
            import age_transfer
            import img_process
            import lm_process
            import torch

            args = type("Args", (), {"img_size": self.cfg.img_size, "group": self.cfg.n_groups,
                                      "batch_size": self.cfg.batch_size, "snapshot": self.cfg.snapshot})()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = age_transfer.Model(args)
            model.G_model.to(device)
            model.G_model.load_state_dict(torch.load(self.cfg.snapshot, map_location=device)["g_ema"])
            self._model = (model, lm_process.LandmarkProcessing(), img_process.ImageProcessing(), device)
        return self._model

    def age_image(self, image_path: Path, out_dir: Path) -> list[AgedImage]:
        """Write one aged PNG per target age group and return the base -> aged mapping."""
        import cv2
        from PIL import Image
        from torchvision import transforms

        model, landmarks, imgproc, device = self._load()
        img = cv2.imread(str(image_path))
        lm = landmarks.detector(img)
        if lm is None:
            return []
        face = Image.fromarray(cv2.cvtColor(imgproc.crop(img, lm), cv2.COLOR_BGR2RGB))
        to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3)])
        outputs = model.generate_image(to_tensor(face).unsqueeze(0).to(device), self.cfg.n_groups,
                                       return_all=True, interpolate=False)
        out_dir.mkdir(parents=True, exist_ok=True)
        results = []
        for i, tensor in enumerate(outputs or []):
            pic = transforms.ToPILImage()((tensor.cpu().clamp(-1, 1) + 1) / 2)
            name = aged_filename(image_path.stem, i)
            pic.save(out_dir / name)
            results.append(AgedImage(image_path.name, name, age_tag(i)))
        return results
