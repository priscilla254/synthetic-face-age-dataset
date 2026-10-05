from flask import Flask, render_template, request, send_from_directory, redirect, url_for
import os
import csv
import shutil
from pathlib import Path
import glob
from collections import defaultdict

app = Flask(__name__)

# 📁 Folder setup
BASE_FOLDERS = {
    "synthetic": "/home/hmc/pb543/diffusers/examples/dreambooth/synthetic_dataset",
    "AgeTransGAN images": "/home/hmc/pb543/AgeTransGAN_myedit/test/step2_seed_images",
    "aged": "/home/hmc/pb543/file_browser/aged_images"
}
DEFAULT_FOLDER = "synthetic"

# 👶 Age group mapping
AGE_GROUPS = {
    "Minors": "13-17",
    "Threshold": "18-20",
    "Legal": "21-25",
    "Adults": "26-40",
    "Older": "41plus"
}

# 🧾 Load metadata
def load_metadata(csv_path):
    metadata = {}
    try:
        with open(csv_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                metadata[row['filename']] = row.get('prompt', '')
    except FileNotFoundError:
        pass
    return metadata

# 📄 Index route
@app.route("/", methods=["GET"])
def index():
    folder_key = request.args.get("folder", DEFAULT_FOLDER)
    path = request.args.get("path", "")
    page = int(request.args.get("page", 1))
    per_page = 25

    base_dir = BASE_FOLDERS.get(folder_key)
    if not base_dir:
        return render_template("index.html", error="Invalid folder selection", files=[], path=path, page=page, has_next=False, folder=folder_key)

    full_path = os.path.join(base_dir, path)
    if not os.path.exists(full_path):
        return render_template("index.html", error="Path does not exist", files=[], path=path, page=page, has_next=False, folder=folder_key)

    ethnicity_options = []
    age_group_options = []

    if folder_key in ["aged", "AgeTransGAN images"]:
        ethnicity_options = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
        if path:
            split = path.strip("/").split("/")
            if len(split) == 1:
                age_group_base = os.path.join(base_dir, split[0])
                if os.path.exists(age_group_base):
                    age_group_options = sorted([d for d in os.listdir(age_group_base) if os.path.isdir(os.path.join(age_group_base, d))])

    metadata = load_metadata(os.path.join(full_path, "metadata.csv"))
    all_files = sorted(os.listdir(full_path))
    all_files = [f for f in all_files if f.lower() != "metadata.csv"]

    files_info = []
    for f in all_files:
        abs_path = os.path.join(full_path, f)
        is_image = f.lower().endswith((".png", ".jpg", ".jpeg"))
        subpath = os.path.join(path, f)

        file_info = {
            "name": f,
            "is_dir": os.path.isdir(abs_path),
            "is_image": is_image,
            "subpath": subpath,
            "prompt": metadata.get(f, "")
        }

        if is_image:
            identity_prefix = f.split("_")[0]
            ethnicity = Path(subpath).parts[0] if len(Path(subpath).parts) > 0 else "Unknown"
            search_pattern = os.path.join(BASE_FOLDERS["aged"], ethnicity, "*", f"{identity_prefix}_*.png")
            existing_variants = glob.glob(search_pattern)
            file_info["disable_move"] = len(existing_variants) >= 2
        else:
            file_info["disable_move"] = False

        files_info.append(file_info)

    # 🔶 Highlight files with shared identity
    identity_map = defaultdict(list)
    for f in files_info:
        if f["is_image"]:
            prefix = f["name"].split("_")[0]
            identity_map[prefix].append(f["name"])
    for f in files_info:
        prefix = f["name"].split("_")[0]
        f["highlight"] = len(identity_map[prefix]) > 1

    start = (page - 1) * per_page
    end = start + per_page
    paginated_files = files_info[start:end]
    has_next = end < len(files_info)

    return render_template("index.html", files=paginated_files, path=path, page=page, has_next=has_next, folder=folder_key,
                           age_groups=AGE_GROUPS, ethnicity_options=ethnicity_options, age_group_options=age_group_options)

# 🖼️ Serve images
@app.route("/output/<path:filename>")
def serve_file(filename):
    folder_key = request.args.get("folder", DEFAULT_FOLDER)
    base_dir = BASE_FOLDERS.get(folder_key, BASE_FOLDERS[DEFAULT_FOLDER])
    full_dir = os.path.join(base_dir, os.path.dirname(filename))
    file_name = os.path.basename(filename)
    return send_from_directory(full_dir, file_name)

# 🗑️ Delete files
@app.route("/delete", methods=["POST"])
def delete_file():
    folder = request.form.get("folder", DEFAULT_FOLDER)
    file_to_delete = request.form.get("file_path")
    current_path = request.form.get("current_path", "")
    page = request.form.get("page", 1)

    base_dir = BASE_FOLDERS.get(folder, BASE_FOLDERS[DEFAULT_FOLDER])
    abs_path = os.path.abspath(os.path.join(base_dir, file_to_delete))

    if not abs_path.startswith(os.path.abspath(base_dir)):
        return "Invalid delete path", 403

    if os.path.exists(abs_path) and os.path.isfile(abs_path):
        os.remove(abs_path)

        metadata_dir = os.path.dirname(abs_path)
        metadata_path = os.path.join(metadata_dir, "metadata.csv")
        if os.path.exists(metadata_path):
            filename_only = os.path.basename(file_to_delete)
            with open(metadata_path, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                rows = list(reader)
                fieldnames = reader.fieldnames

            updated_rows = [row for row in rows if os.path.basename(row.get("filename", "")) != filename_only]

            with open(metadata_path, "w", newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                if updated_rows:
                    writer.writerows(updated_rows)

    return redirect(url_for('index', path=current_path, folder=folder, page=page))

# 🧠 Assign age group
@app.route("/assign_age_group", methods=["POST"])
def assign_age_group():
    age_key = request.form.get("age_group")
    file_path = request.form.get("file_path")
    current_path = request.form.get("current_path", "")
    folder = request.form.get("folder", "aged")
    page = request.form.get("page", 1)

    age_folder = AGE_GROUPS.get(age_key)
    if not age_folder:
        return "Invalid age group", 400

    source_base_dir = BASE_FOLDERS.get(folder)
    dest_base_dir = BASE_FOLDERS["aged"]
    if not source_base_dir or not dest_base_dir:
        return "Invalid folder", 400

    src_path = os.path.join(source_base_dir, file_path)
    filename = os.path.basename(file_path)
    identity_prefix = filename.split("_")[0]

    relative_parts = Path(file_path).parts
    ethnicity = relative_parts[0] if len(relative_parts) >= 1 else "Unknown"

    search_pattern = os.path.join(dest_base_dir, ethnicity, "*", f"{identity_prefix}_*.png")
    existing_variants = glob.glob(search_pattern)
    if len(existing_variants) >= 2:
        return redirect(url_for('index', path=current_path, folder=folder, page=page))

    dest_dir = os.path.join(dest_base_dir, ethnicity, age_folder)
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, filename)

    try:
        shutil.move(src_path, dest_path)
        from clean_up import cleanup_and_relabel
        cleanup_and_relabel(dest_dir)

    except Exception as e:
        return f"Failed to move file: {str(e)}", 500

    return redirect(url_for('index', path=current_path, folder=folder, page=page))

if __name__ == "__main__":
    app.run(debug=True)
