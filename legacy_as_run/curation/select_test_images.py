import os
import shutil
import pandas as pd
import random

# Base folders
source_base = 'aged_images'
test_folder = 'test'

# Number of images to select per age group subfolder
num_images_per_subfolder = 1

# Create test folder if it doesn't exist
os.makedirs(test_folder, exist_ok=True)

# Temporary storage for copied image paths and metadata
collected_data = []

# Walk through each ethnicity and age group subfolder
for ethnicity in os.listdir(source_base):
    ethnicity_path = os.path.join(source_base, ethnicity)
    if os.path.isdir(ethnicity_path):
        for age_group in os.listdir(ethnicity_path):
            age_group_path = os.path.join(ethnicity_path, age_group)
            if os.path.isdir(age_group_path):
                
                metadata_path = os.path.join(age_group_path, 'metadata.csv')
                if not os.path.exists(metadata_path):
                    continue
                
                metadata_df = pd.read_csv(metadata_path)
                image_files = [f for f in os.listdir(age_group_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                if not image_files:
                    continue
                
                selected_images = random.sample(image_files, min(num_images_per_subfolder, len(image_files)))
                
                for image_file in selected_images:
                    src_image_path = os.path.join(age_group_path, image_file)
                    image_metadata = metadata_df[metadata_df['filename'] == image_file]
                    if not image_metadata.empty:
                        collected_data.append({
                            'src_path': src_image_path,
                            'original_metadata': image_metadata.iloc[0].to_dict()
                        })

# Shuffle the collected data
random.shuffle(collected_data)

# Copy images and assign new filename and id
final_metadata_list = []
for idx, item in enumerate(collected_data, start=1):
    ext = os.path.splitext(item['src_path'])[-1].lower()
    new_filename = f"img{idx:03d}{ext}"
    dest_path = os.path.join(test_folder, new_filename)
    
    shutil.copy2(item['src_path'], dest_path)
    
    updated_metadata = item['original_metadata']
    updated_metadata['filename'] = new_filename
    updated_metadata['id'] = idx  # Overwrite the old ID with a new one
    final_metadata_list.append(updated_metadata)

# Save final metadata
if final_metadata_list:
    final_metadata_df = pd.DataFrame(final_metadata_list)
    final_metadata_path = os.path.join(test_folder, 'metadata.csv')
    final_metadata_df.to_csv(final_metadata_path, index=False)

print(f"{len(final_metadata_list)} images and metadata copied, renamed, and reindexed in '{test_folder}' successfully.")
