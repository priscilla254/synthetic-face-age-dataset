"""Stage 1: demographically varied seed portraits with Stable Diffusion XL.

``PromptBuilder`` is pure Python (no GPU) and unit-tested. ``SeedGenerator``
wraps the diffusers pipeline and is a straight restructuring of
``legacy_as_run/generation/1.generate_faces.py``; it has not been re-run.
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path

HAIRSTYLES_AFRICAN_WOMEN = ["braids", "cornrows", "afro", "weaves", "straight hair", "wavy hair", "short hair"]
HAIRSTYLES_OTHER_WOMEN = ["long ponytails", "fringe cuts", "pixie cut", "bob", "long wavy hair", "straight hair", "dyed"]
HAIRSTYLES_MEN = ["buzz cut", "crew cut", "textured crop", "side part", "slicked back", "quiff"]

ACCESSORIES_FEMALE = ["wearing makeup", "wearing prescription glasses", "wearing sunglasses",
                      "has ear piercings", "has a nose piercing", "has a nose-to-ear chain"]
ACCESSORIES_MALE = ["wearing prescription glasses", "wearing sunglasses",
                    "has a visible tattoo", "has an eyebrow piercing"]

BACKGROUND_LIGHTING = {
    "a neutral indoor background": ["soft warm indoor lighting", "white fluorescent lighting"],
    "a supermarket store": ["harsh overhead store lighting", "white fluorescent lighting"],
    "a liquor store": ["harsh overhead store lighting", "white fluorescent lighting"],
    "a bedroom": ["soft warm indoor lighting", "colored lighting effects from a TV or neon signs"],
    "a living room at home": ["soft warm indoor lighting", "colored lighting effects from a TV or neon signs"],
}

NEGATIVE_PROMPT = (
    "cartoon, anime, illustration, painting, sketch, unrealistic, fake skin, plastic texture, vector art, "
    "fantasy, 2D, bad anatomy, distorted face, asymmetry, blurry, low quality, artifact, glitch, "
    "oversaturated colors, unnatural lighting, glowing eyes, deformed ears, missing facial features, "
    "poor composition, text artifacts, watermark, fashion model, glamour, perfect skin, retouched, beauty shot, "
    "magazine photo, staged, unrealistic body, idealized body, editorial style, seductive pose, runway look"
)

EXPRESSION = "a neutral expression with mouth closed"
POSE = "facing straight toward the camera, eyes open, head level"
REALISM_KEYWORDS = "natural-looking, everyday person, not a fashion model, realistic proportions"


@dataclass
class SeedAttributes:
    """Everything that determines one seed portrait (and goes into the metadata)."""

    ethnicity: str
    gender: str
    age_group: str
    hairstyle: str
    facial_hair: str
    accessories: list[str]
    outfit: str
    background: str
    lighting: str
    body_type: str
    prompt: str = ""

    def to_row(self) -> dict:
        row = asdict(self)
        row["accessories"] = ", ".join(self.accessories)
        return row


class PromptBuilder:
    """Randomised, rule-based prompt construction (behaviour of the original script).

    Two quirks of the original are kept deliberately so results stay explainable:
    ethnicity rules match on substrings (``"african" in ethnicity.lower()`` does
    not match "Black Caribbean"), and the sampled ``image_quality`` was never
    used in the prompt, so it is not part of this class.
    """

    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random()

    def sample(self, ethnicity: str, gender: str, age_group: str) -> SeedAttributes:
        rng, eth = self.rng, ethnicity.lower()
        if "african" in eth and gender == "female":
            hairstyle = rng.choice(HAIRSTYLES_AFRICAN_WOMEN)
        else:
            hairstyle = rng.choice(HAIRSTYLES_OTHER_WOMEN if gender == "female" else HAIRSTYLES_MEN)
        facial_hair = "has facial hair" if gender == "male" and rng.random() < 0.5 else ""

        pool = ACCESSORIES_FEMALE if gender == "female" else ACCESSORIES_MALE
        accessories = rng.sample(pool, k=rng.choices([0, 1, 2], weights=[0.4, 0.4, 0.2])[0])
        if eth == "arabic" and gender == "female":
            accessories.append("wearing a hijab")
        if "african" in eth and gender == "male" and rng.random() < 0.3:
            accessories.append("wearing a durag")
        if eth == "indian" and gender == "male" and rng.random() < 0.25:
            accessories.append("wearing an Indian-style turban")

        if age_group in {"13", "14", "15", "16", "17"} and rng.random() < 0.3:
            outfit = "a school uniform"
        elif age_group in {"13", "14", "15", "16", "17"}:
            outfit = rng.choice(["casual clothes", "a hoodie", "sportswear"])
        else:
            outfit = rng.choice(["business attire", "casual clothes", "evening wear", "sportswear"])

        background = rng.choice(list(BACKGROUND_LIGHTING))
        lighting = rng.choice(BACKGROUND_LIGHTING[background])
        body_type = "plus-size" if rng.random() < 0.25 else "average build"

        attrs = SeedAttributes(ethnicity, gender, age_group, hairstyle, facial_hair,
                               accessories, outfit, background, lighting, body_type)
        attrs.prompt = self.render(attrs)
        return attrs

    @staticmethod
    def render(a: SeedAttributes) -> str:
        traits = ", ".join(t for t in (a.hairstyle, a.facial_hair) if t)
        accessory_str = ", " + ", ".join(a.accessories) if a.accessories else ""
        return (
            f"A {a.age_group}-year-old A portrait of a {a.body_type} {a.ethnicity} {a.gender} "
            f"with {traits}{accessory_str}. Wearing {a.outfit}, {POSE}, {EXPRESSION}. "
            f"Set in {a.background}, under {a.lighting}. "
            f"{REALISM_KEYWORDS}, photorealistic, 1024x1024 resolution, shallow depth of field."
        )


class SeedGenerator:
    """Generate and save seed images; one metadata row per image (incl. the seed)."""

    def __init__(self, config, builder: PromptBuilder | None = None):
        self.cfg = config.seed_generation
        self.builder = builder or PromptBuilder(random.Random(config.random_seed))
        self._pipe = None

    def _pipeline(self):
        if self._pipe is None:  # heavy imports only when actually generating
            import torch
            from diffusers import StableDiffusionXLPipeline

            self._pipe = StableDiffusionXLPipeline.from_pretrained(
                self.cfg.model_id, torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
            ).to("cuda")
        return self._pipe

    def generate(self, ethnicity: str, out_dir: Path, start_index: int = 1) -> list[dict]:
        import torch

        out_dir.mkdir(parents=True, exist_ok=True)
        genders = ["female"] * self.cfg.images_per_gender + ["male"] * self.cfg.images_per_gender
        self.builder.rng.shuffle(genders)
        rows, index = [], start_index
        for gender in genders:
            for age in self.cfg.prompt_ages:
                attrs = self.builder.sample(ethnicity, gender, age)
                seed = self.builder.rng.randint(0, 2**32 - 1)  # logged, unlike the original run
                image = self._pipeline()(
                    prompt=attrs.prompt, negative_prompt=NEGATIVE_PROMPT,
                    num_inference_steps=self.cfg.num_inference_steps,
                    guidance_scale=self.cfg.guidance_scale,
                    generator=torch.manual_seed(seed),
                    height=self.cfg.height, width=self.cfg.width,
                ).images[0]
                name = f"img{index:03d}.jpg"
                image.save(out_dir / name)
                rows.append({"filename": name, "seed": seed, **attrs.to_row()})
                index += 1
        return rows
