import os
import random
import pandas as pd
import streamlit as st
from datetime import datetime

# === Config ===
image_dir = os.environ.get("FACEDATA_HUMAN_EVAL_DIR", "data/images/human_eval_subset")
metadata_path = os.path.join(image_dir, "human_eval_subset_metadata.csv")
results_csv = "evaluation_results.csv"
IMAGES_PER_USER = 30

# === Load metadata ===
df = pd.read_csv(metadata_path)

# === UI: Title and Login ===
st.title("🧠 Human Evaluation of Synthetic Faces")
user_id = st.text_input("🔐 Enter your name:", max_chars=30)

# === Proceed once user ID is entered ===
if user_id:
    st.session_state.setdefault("user_id", user_id)

    if "user_images" not in st.session_state:
        all_filenames = df["filename"].tolist()
        random.shuffle(all_filenames)
        st.session_state.user_images = all_filenames[:IMAGES_PER_USER]
        st.session_state.current_index = 0

    if st.session_state.current_index >= IMAGES_PER_USER:
        st.success("✅ You’ve completed all 30 images. Thank you!")
        st.progress(100)
        st.stop()

    # === Get current image and metadata ===
    current_filename = st.session_state.user_images[st.session_state.current_index]
    row = df[df["filename"] == current_filename].iloc[0]
    image_path = os.path.join(image_dir, current_filename)

    # === Progress bar and counter ===
    progress = (st.session_state.current_index + 1) / IMAGES_PER_USER
    st.progress(progress)
    st.markdown(f"**Image {st.session_state.current_index + 1} of {IMAGES_PER_USER}**")

    # === Layout ===
    left_col, right_col = st.columns([1, 1.2])

    with left_col:
        if os.path.exists(image_path):
            st.image(image_path, width=300)
            st.markdown(f"**Ethnicity:** {row['ethnicity']}")
            st.markdown(f"**Age Group:** {row['age_group']}")
        else:
            st.error(f"❌ Image not found: {current_filename}")
            st.stop()

    with right_col:
        realism = st.slider("1️⃣ Realism", 1, 5, 3, help="Does this look like a real human?")
        age_match = st.slider("2️⃣ Age Appropriateness", 1, 5, 3, help="Does this match the labeled age group?")
        ethnicity_match = st.slider("3️⃣ Ethnic Consistency", 1, 5, 3, help="Does this match the labeled ethnicity?")

        submitted = st.button("✅ Submit Rating")

        if submitted:
            result = {
                "user_id": st.session_state.user_id,
                "filename": current_filename,
                "ethnicity": row["ethnicity"],
                "age_group": row["age_group"],
                "realism": realism,
                "age_appropriateness": age_match,
                "ethnic_consistency": ethnicity_match,
                "timestamp": datetime.now().isoformat()
            }

            if os.path.exists(results_csv):
                pd.DataFrame([result]).to_csv(results_csv, mode='a', header=False, index=False)
            else:
                pd.DataFrame([result]).to_csv(results_csv, index=False)

            st.session_state.current_index += 1
            st.rerun()
