import os
import shutil
import csv
import re

# === CONFIG ===
aged_flat_folder = '/home/hmc/pb543/AgeTransGAN_myedit/test/step2_seed_images'
output_base_folder = 'aged_images'
original_metadata_path = '/home/hmc/pb543/diffusers/examples/dreambooth/synthetic_dataset2/step1_seed_images/metadata.csv'

# Age group mapping from AgeTransGAN label (ageX) to folder
age_map = {
    'age1': '13-17',
    'age2': '18-20',
    'age3': '21-25',
    'age4': '26-40',
    'age5': '41plus'
}

# Create output folders
for age_group in age_map.values():
    os.makedirs(os.path.join(output_base_folder, age_group), exist_ok=True)

# Load original metadata
original_metadata = {}
with open(original_metadata_path, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        original_metadata[row['filename']] = row  # Store entire row to access multiple fields

# Prepare structure to hold grouped metadata
grouped_metadata = {age_group: [] for age_group in age_map.values()}

# Process each image
for filename in os.listdir(aged_flat_folder):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    src_path = os.path.join(aged_flat_folder, filename)

    # Match age tag in filename (e.g., age1, age3)
    age_tag_match = re.search(r'age[1-5]', filename)
    if not age_tag_match:
        print(f"⚠️ Skipped (no age tag): {filename}")
        continue

    age_tag = age_tag_match.group()
    age_group = age_map.get(age_tag)

    if not age_group:
        print(f"⚠️ Unknown age tag: {filename}")
        continue

    # Extract base ID (e.g., img025 from img025_age3.png)
    original_name_match = re.match(r'(img\d+)', filename)
    if not original_name_match:
        print(f"⚠️ Could not extract base name: {filename}")
        continue

    original_filename = original_name_match.group(1) + '.jpg'  # e.g., img025.jpg

    # Get metadata values
    original_data = original_metadata.get(original_filename, {})
    ethnicity = original_data.get('ethnicity', 'Unknown')
    gender = original_data.get('gender', 'Unknown')

    # Rename file to original style (img025.jpg) for output
    new_filename = original_filename
    dst_path = os.path.join(output_base_folder, age_group, new_filename)
    shutil.copyfile(src_path, dst_path)

    # Save metadata
    grouped_metadata[age_group].append({
        'filename': new_filename,
        'age_range': age_group,
        'ethnicity': ethnicity,
        'gender': gender
    })

# Write metadata.csv for each age group
for age_group, records in grouped_metadata.items():
    metadata_path = os.path.join(output_base_folder, age_group, 'metadata.csv')
    with open(metadata_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'age_range', 'ethnicity', 'gender'])
        writer.writeheader()
        writer.writerows(records)
    print(f"✅ metadata.csv written for {age_group}")

print("🎉 All images sorted, renamed, and metadata generated.")
