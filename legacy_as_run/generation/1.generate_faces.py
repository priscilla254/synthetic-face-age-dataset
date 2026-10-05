import os
import random
import torch
import pandas as pd
from pathlib import Path
from PIL import Image
from diffusers import StableDiffusionXLPipeline

# ==== SETUP ====

base_dir = Path("synthetic_dataset")
if not base_dir.exists():
    base_dir.mkdir(parents=True, exist_ok=True)

# Load Stable Diffusion XL model
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
).to("cuda")

# ==== Configs ====
age_groups = ["13", "17", "25"]
ethnicity = "Black Caribbean"  # Change to desired ethnicity
ethnicity_dir = base_dir / ethnicity
ethnicity_dir.mkdir(parents=True, exist_ok=True)
metadata_path = ethnicity_dir / "metadata.csv"
metadata_list = []

# Start fresh index or continue from existing
if metadata_path.exists():
    existing = pd.read_csv(metadata_path)
    next_index = existing["id"].str.extract(r"img(\d+)", expand=False).dropna().astype(int).max().values[0] + 1
else:
    next_index = 1

# Gender distribution
genders = ['female'] * 20 + ['male'] * 20
random.shuffle(genders)

# Definitions
hairstyles_african_women=["braids", "cornrows", "afro", "weaves", "straight hair", "wavy hair", "short hair"]
hairstyles_others_women = ["long ponytails", "fringe cuts", "pixie cut", "bob", "long wavy hair", "straight hair", "dyed"]
hairstyles_others_men = ["buzz cut", "crew cut", "textured crop", "side part", "slicked back", "quiff"]

accessories_female = [
    "wearing makeup", "wearing prescription glasses", "wearing sunglasses",
    "has ear piercings", "has a nose piercing", "has a nose-to-ear chain", "wearing a hijab"
]

accessories_male = [
    "wearing prescription glasses", "wearing sunglasses",
    "has a visible tattoo", "has an eyebrow piercing",
    "wearing a durag", "wearing an Indian-style turban"
]

background_lighting_map = {
    "a neutral indoor background": ["soft warm indoor lighting", "white fluorescent lighting"],
    "a supermarket store": ["harsh overhead store lighting", "white fluorescent lighting"],
    "a liquor store": ["harsh overhead store lighting", "white fluorescent lighting"],
    "a bedroom": ["soft warm indoor lighting", "colored lighting effects from a TV or neon signs"],
    "a living room at home": ["soft warm indoor lighting", "colored lighting effects from a TV or neon signs"]
}

image_qualities = ["low resolution", "high definition", "slightly blurred", "camera focused"]
expression = "a neutral expression with mouth closed"
pose = "facing straight toward the camera, eyes open, head level"

def truncate_prompt(prompt, max_tokens=77):
    return ' '.join(prompt.split()[:max_tokens])

# ==== Image Generation Loop ====
for gender in genders:
    for age_group in age_groups:
        # === Hairstyle selection ===
        if "african" in ethnicity.lower() and gender == "female":
            hairstyle = random.choice(hairstyles_african_women)
        else:
            hairstyle = random.choice(hairstyles_others_women if gender == "female" else hairstyles_others_men)

        facial_hair = "has facial hair" if gender == "male" and random.random() < 0.5 else ""

       # === Accessory selection with filtering ===
        if gender == "female":
            accessory_pool = [a for a in accessories_female if a != "wearing a hijab"]
        else:
            accessory_pool = [a for a in accessories_male if a not in ["wearing a durag", "wearing an Indian-style turban"]]

        # Sample from the filtered pool
        accessories = random.sample(accessory_pool, k=random.choices([0, 1, 2], weights=[0.4, 0.4, 0.2])[0])

        # Arabic female: always wear a hijab
        if ethnicity.lower() == "arabic" and gender == "female":
            accessories.append("wearing a hijab")

        # African male: 30% chance to wear a durag
        if "african" in ethnicity.lower() and gender == "male":
            if random.random() < 0.3:
                accessories.append("wearing a durag")

        # Indian male: 25% chance to wear Indian-style turban
        if ethnicity.lower() == "indian" and gender == "male":
            if random.random() < 0.25:
                accessories.append("wearing an Indian-style turban")


        # === Outfit and context ===
        outfit = random.choice(["business attire", "casual clothes", "evening wear", "sportswear"])
        if age_group in ["13", "14", "15", "16", "17"]:
            outfit = "a school uniform" if random.random() < 0.3 else random.choice(["casual clothes", "a hoodie", "sportswear"])

        background = random.choice(list(background_lighting_map.keys()))
        lighting = random.choice(background_lighting_map[background])
        image_quality = random.choice(image_qualities)

        # === Prompt construction ===
        traits = [hairstyle, facial_hair if facial_hair else None]
        trait_str = ", ".join(filter(None, traits))
        accessory_str = ", " + ", ".join(accessories) if accessories else ""

        subject_id = f"img{next_index:03d}"
        filename = f"{subject_id}.jpg"

        body_type = "plus-size" if random.random() < 0.25 else "average build"
        realism_keywords = "natural-looking, everyday person, not a fashion model, realistic proportions"

        prompt = (
            f"A portrait of a {body_type} {ethnicity} {gender} with {trait_str}{accessory_str}. "
            f"Wearing {outfit}, {pose}, {expression}. "
            f"Set in {background}, under {lighting}. "
            f"{realism_keywords}, photorealistic, 1024x1024 resolution, shallow depth of field."
        )
        prompt = f"A {age_group}-year-old " + prompt



        negative_prompt = (
            "cartoon, anime, illustration, painting, sketch, unrealistic, fake skin, plastic texture, vector art, "
            "fantasy, 2D, bad anatomy, distorted face, asymmetry, blurry, low quality, artifact, glitch, "
            "oversaturated colors, unnatural lighting, glowing eyes, deformed ears, missing facial features, "
            "poor composition, text artifacts, watermark, fashion model, glamour, perfect skin, retouched, beauty shot, "
            "magazine photo, staged, unrealistic body, idealized body, editorial style, seductive pose, runway look"
        )

        print(f"📸 Generating image {subject_id} for {ethnicity} {gender}...\nPrompt: {prompt}\n")

        try:
            generator = torch.manual_seed(random.randint(0, 2**32 - 1))
            image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=50,
            guidance_scale=7.5,
            generator=generator,
            height=1024,
            width=1024
            ).images[0]

            image.save(ethnicity_dir / filename)

            metadata_list.append({
            "id": subject_id,
            "filename": filename,
            "ethnicity": ethnicity,
            "gender": gender,
            "age_group": age_group,
            "hairstyle": hairstyle,
            "facial_hair": facial_hair,
            "accessories": ", ".join(accessories),
            "background": background,
            "lighting": lighting,
            "image_quality": image_quality,
            "outfit": outfit,
            "prompt": prompt,
            "status": "pending"
            })

            next_index += 1

        except Exception as e:
            print(f"❌ Failed to generate image {subject_id}: {e}")

# ==== Save Metadata in Ethnicity Subfolder ====
pd.DataFrame(metadata_list).to_csv(metadata_path, index=False)
print(f"✅ {len(metadata_list)} new images saved to '{ethnicity_dir}'")
print(f"📄 Metadata saved to: {metadata_path}")
