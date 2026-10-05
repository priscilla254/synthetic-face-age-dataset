import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# === Config ===
dataset_dir = "all_synthetic_images_restored"
metadata_path = os.path.join(dataset_dir, "all_metadata.csv")
train_dir = os.path.join(dataset_dir, "train")
test_dir = os.path.join(dataset_dir, "test")
train_split = 0.8

# === Step 1: Clean previous train/test folders ===
for folder in [train_dir, test_dir]:
    if os.path.exists(folder):
        shutil.rmtree(folder)

# === Step 2: Load and filter metadata to only existing image files ===
df = pd.read_csv(metadata_path)
existing_files = set(os.listdir(dataset_dir))
df = df[df["filename"].isin(existing_files)].reset_index(drop=True)

# === Step 3: Shuffle metadata ===
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# === Step 4: Rename image files sequentially ===
new_filenames = []
for i, row in df.iterrows():
    old_name = row["filename"]
    new_name = f"img{i+1:03d}.png"
    old_path = os.path.join(dataset_dir, old_name)
    new_path = os.path.join(dataset_dir, new_name)

    if os.path.exists(old_path):
        os.rename(old_path, new_path)
    else:
        print(f"⚠️ Missing image: {old_name}")
    
    new_filenames.append(new_name)

df["filename"] = new_filenames

# === Step 5: Drop 'id' column if it exists ===
if "id" in df.columns:
    df.drop(columns=["id"], inplace=True)

# === Step 6: Create stratification label (ethnicity + age_group) ===
df["strata"] = df["ethnicity"].astype(str) + "_" + df["age_group"].astype(str)

# === Step 7: Stratified train/test split ===
df_train, df_test = train_test_split(
    df,
    train_size=train_split,
    random_state=42,
    stratify=df["strata"]
)

# Drop the helper column
df.drop(columns=["strata"], inplace=True)
df_train.drop(columns=["strata"], inplace=True)
df_test.drop(columns=["strata"], inplace=True)

# === Step 8: Save updated metadata ===
df.to_csv(os.path.join(dataset_dir, "all_metadata.csv"), index=False)
df_train.to_csv(os.path.join(dataset_dir, "train_metadata.csv"), index=False)
df_test.to_csv(os.path.join(dataset_dir, "test_metadata.csv"), index=False)

# === Step 9: Create train/test folders and move images ===
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

for filename in df_train["filename"]:
    src = os.path.join(dataset_dir, filename)
    dst = os.path.join(train_dir, filename)
    if os.path.exists(src):
        shutil.move(src, dst)

for filename in df_test["filename"]:
    src = os.path.join(dataset_dir, filename)
    dst = os.path.join(test_dir, filename)
    if os.path.exists(src):
        shutil.move(src, dst)

print("✅ Done! Dataset shuffled, renamed, stratified, and split into train/test.")
