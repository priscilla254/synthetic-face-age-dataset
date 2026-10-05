import os
import csv
import pandas as pd
import random
import csv

# Set this to the dataset directory you want to clean
DATASET_DIR = "/home/hmc/pb543/file_browser/test"
# DATASET_DIR = "/home/hmc/pb543/diffusers/examples/dreambooth/synthetic_dataset/Black Caribbean"

def cleanup_and_generate_metadata_by_subfolder(directory, ethnicity):
    for subdir in os.listdir(directory):
        full_path = os.path.join(directory, subdir)
        if os.path.isdir(full_path):
            age_group = subdir

            # Try to load existing metadata to retain gender labels
            metadata_path = os.path.join(full_path, "metadata.csv")
            existing_gender_map = {}

            if os.path.exists(metadata_path):
                with open(metadata_path, newline='') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        existing_gender_map[row["filename"]] = row.get("gender", "Unknown")

            # Reorder and rename image files
            image_files = sorted([
                f for f in os.listdir(full_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ])

            new_rows = []
            for idx, old_filename in enumerate(image_files, start=1):
                ext = os.path.splitext(old_filename)[1].lower()
                new_filename = f"img{idx:03d}{ext}"

                old_path = os.path.join(full_path, old_filename)
                new_path = os.path.join(full_path, new_filename)
                os.rename(old_path, new_path)

                gender = existing_gender_map.get(old_filename, "Unknown")

                new_rows.append({
                    "id": idx,
                    "filename": new_filename,
                    "ethnicity": ethnicity,
                    "age_group": age_group,
                    "gender": gender
                })

            # Save updated metadata
            with open(metadata_path, "w", newline='') as csvfile:
                fieldnames = ["id", "filename", "ethnicity", "age_group", "gender"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(new_rows)

            print(f"✅ Renamed and updated metadata for '{age_group}' with {len(new_rows)} images.")
def cleanup_and_relabel(directory):
    ethnicity = os.path.basename(os.path.dirname(directory))
    age_group = os.path.basename(directory)

    # Load gender info from AgeTransGAN metadata
    gender_map = {}
    seed_metadata_path = os.path.join("AgeTransGAN images", ethnicity, "metadata.csv")
    if os.path.exists(seed_metadata_path):
        df = pd.read_csv(seed_metadata_path)
        gender_map = dict(zip(df["filename"], df["gender"]))

    image_files = sorted([
        f for f in os.listdir(directory)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    new_rows = []
    for idx, filename in enumerate(image_files, start=1):
        ext = os.path.splitext(filename)[1].lower()
        new_filename = f"img{idx:03d}{ext}"

        old_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_filename)
        os.rename(old_path, new_path)

        gender = gender_map.get(filename, "Unknown")

        new_rows.append({
            "id": idx,
            "filename": new_filename,
            "ethnicity": ethnicity,
            "age_group": age_group,
            "gender": gender
        })

    metadata_path = os.path.join(directory, "metadata.csv")
    with open(metadata_path, "w", newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["id", "filename", "ethnicity", "age_group", "gender"])
        writer.writeheader()
        writer.writerows(new_rows)

    print(f"✅ [Flask] Cleaned and updated metadata in {directory}")
# === MAIN ENTRY ===
if __name__ == "__main__":
    ethnicity = os.path.basename(DATASET_DIR)
    if "aged_images" in DATASET_DIR:
        cleanup_and_generate_metadata_by_subfolder(DATASET_DIR, ethnicity)
    else:
        print("❌ Unrecognized dataset type. Make sure the path contains 'aged_images'.")
