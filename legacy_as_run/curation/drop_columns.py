import pandas as pd
import os


df = pd.read_csv("all_synthetic_images_restored/all_metadata.csv")
df = df.drop(columns=["id"])
df.to_csv("all_synthetic_images_restored/all_metadata.csv", index=False)
