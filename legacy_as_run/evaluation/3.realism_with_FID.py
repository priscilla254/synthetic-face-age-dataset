import os
import random
import shutil
import zipfile
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
import numpy as np
from scipy.linalg import sqrtm

# Load InceptionV3 model pre-trained on ImageNet
model_incep = InceptionV3(include_top=False, weights='imagenet', pooling='avg')

def get_inception_features(img_path):
    try:
        # Load image and resize to 299x299 for InceptionV3 input
        img = image.load_img(img_path, target_size=(299, 299))
        img_array = image.img_to_array(img)  # Convert image to numpy array
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = preprocess_input(img_array)  # Preprocess for InceptionV3
        features = model_incep.predict(img_array)  # Get feature vector
        return features
    except Exception as e:
        print(f"Error processing image {img_path}: {e}")
        return None

def calculate_fid(real_images, synthetic_images):
    real_features = []
    synthetic_features = []

    print(f"Processing {len(real_images)} real images...")
    # Extract features for all real images
    for img_path in real_images:
        features = get_inception_features(img_path)
        if features is not None:
            real_features.append(features)
        else:
            print(f"Skipping real image {img_path} due to error.")
    
    print(f"Processing {len(synthetic_images)} synthetic images...")
    # Extract features for all synthetic images
    for img_path in synthetic_images:
        features = get_inception_features(img_path)
        if features is not None:
            synthetic_features.append(features)
        else:
            print(f"Skipping synthetic image {img_path} due to error.")
    
    # Debug: Print the length of the feature lists
    print(f"Real features count: {len(real_features)}")
    print(f"Synthetic features count: {len(synthetic_features)}")
    
    # Ensure there are valid features before stacking
    if len(real_features) == 0 or len(synthetic_features) == 0:
        raise ValueError("No valid features extracted from images.")
    
    # Convert the list of arrays into a 2D array (samples x features)
    real_features = np.vstack(real_features)  # Stack the features for real images
    synthetic_features = np.vstack(synthetic_features)  # Stack the features for synthetic images

    # Calculate the mean and covariance of the features
    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    mu_fake = np.mean(synthetic_features, axis=0)
    sigma_fake = np.cov(synthetic_features, rowvar=False)

    # Compute FID score
    diff = mu_real - mu_fake
    covmean = sqrtm(sigma_real.dot(sigma_fake))  # Removed unpacking, now only one return value
    fid = np.sum(diff**2) + np.trace(sigma_real + sigma_fake - 2*covmean)
    return fid


# Dataset folder paths 
real_dataset_folder = 'fairface_dataset' #UTKFace
synthetic_folder = 'shuffled_dataset_generalised'

# Get all images in real dataset folder and random sample 1000 images
real_images = [os.path.join(real_dataset_folder, f) for f in os.listdir(real_dataset_folder) if f.endswith('.jpg')]
print(f"Found {len(real_images)} images in real dataset folder.")
sampled_real_images = random.sample(real_images, 1000)

# Select the first 1000 synthetic images
synthetic_images = [os.path.join(synthetic_folder, f) for f in os.listdir(synthetic_folder) if f.endswith(('.png', '.jpg'))]
print(f"Found {len(synthetic_images)} images in synthetic dataset folder.")
sampled_synthetic_images = synthetic_images[:1000]

# Debug: Print the paths of some sampled images
print(f"Sampled real images: {sampled_real_images[:5]}")  # Print first 5 image paths for debug
print(f"Sampled synthetic images: {sampled_synthetic_images[:5]}")  # Print first 5 image paths for debug

# Function to create a directory and copy sampled images there
def copy_images_to_dir(image_paths, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for img_path in image_paths:
        shutil.copy(img_path, output_dir)

# Create directories to store the sampled images
real_images_dir = 'sampled_real_images'
synthetic_images_dir = 'sampled_synthetic_images'

# Copy sampled images to these directories
copy_images_to_dir(sampled_real_images, real_images_dir)
copy_images_to_dir(sampled_synthetic_images, synthetic_images_dir)

# Calculate FID score for the sampled images
fid_score = calculate_fid(sampled_real_images, sampled_synthetic_images)
print(f"FID Score (for 1000 sampled images from real dataset and synthetic dataset): {fid_score}")

# Function to zip a directory for download
def zip_directory(directory_path, zip_filename):
    shutil.make_archive(zip_filename, 'zip', directory_path)

# Zip the directories containing the sampled images
zip_directory(real_images_dir, 'sampled_real_images')
zip_directory(synthetic_images_dir, 'sampled_synthetic_images')

print("Sampled images from real dataset and synthetic dataset have been zipped.")
