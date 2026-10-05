import os
import shutil
import pandas as pd

base_dir = "aged_images2"
output_dir = "all_synthetic_images"
os.makedirs(output_dir, exist_ok=True)

metadata_list = []
image_counter = 1

for ethnicity in os.listdir(base_dir):
    ethnicity_path = os.path.join(base_dir, ethnicity)
    if not os.path.isdir(ethnicity_path):
        continue

    for age_group in os.listdir(ethnicity_path):
        age_group_path = os.path.join(ethnicity_path, age_group)
        if not os.path.isdir(age_group_path):
            continue

        metadata_file = os.path.join(age_group_path, "metadata.csv")
        if not os.path.exists(metadata_file):
            continue

        df = pd.read_csv(metadata_file)
        df["ethnicity"] = ethnicity
        df["age_group"] = age_group

        new_filenames = []
        for old_filename in df["filename"]:
            src = os.path.join(age_group_path, old_filename)
            if not os.path.exists(src):
                continue

            new_filename = f"img{image_counter:03d}.png"
            dst = os.path.join(output_dir, new_filename)
            shutil.copy2(src, dst)
            new_filenames.append(new_filename)
            image_counter += 1

        df["filename"] = new_filenames
        metadata_list.append(df)

# Save combined metadata
if metadata_list:
    full_metadata = pd.concat(metadata_list, ignore_index=True)
    full_metadata.to_csv(os.path.join(output_dir, "all_metadata.csv"), index=False)

print("✅ All images copied, renamed, and metadata updated.")
