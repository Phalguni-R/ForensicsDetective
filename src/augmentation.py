import cv2
import numpy as np
import os
import random
from PIL import Image

def augment_image(img_path, output_folder):
    # Load image in grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error loading {img_path}")
        return
    
    base_name = os.path.basename(img_path).split('.')[0]
    
    # 1. Gaussian Noise (sigma between 5 and 20)
    sigma = random.uniform(5, 20)
    gauss = np.random.normal(0, sigma, img.shape)
    noisy = np.clip(img + gauss, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_folder, f"{base_name}_noise.png"), noisy)
    
    # 2. JPEG Compression (quality between 20 and 80)
    quality = random.randint(20, 80)
    pil_img = Image.fromarray(img)
    pil_img.save(os.path.join(output_folder, f"{base_name}_jpeg.jpg"), "JPEG", quality=quality)
    
    # 3. DPI Downsampling (300 to 150 or 72)
    # Using 0.5x scale for 150 DPI and 0.24x for 72 DPI
    scale = random.choice([0.5, 0.24])
    h, w = img.shape
    downsampled = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(output_folder, f"{base_name}_dpi.png"), downsampled)
    
    # 4. Random Cropping (Remove 1-3% from each border)
    h, w = img.shape
    p = random.uniform(0.01, 0.03)
    top, bottom = int(h * p), int(h * (1 - p))
    left, right = int(w * p), int(w * (1 - p))
    cropped = img[top:bottom, left:right]
    cv2.imwrite(os.path.join(output_folder, f"{base_name}_crop.png"), cropped)
    
    # 5. Bit-Depth Reduction (8-bit to 4-bit grayscale)
    # 4-bit has 16 levels of gray
    bit4 = (img // 16) * 16
    cv2.imwrite(os.path.join(output_folder, f"{base_name}_bit.png"), bit4.astype(np.uint8))

if __name__ == "__main__":
    # Folders based on Task 7.1 organization
    input_dir = "data/original_pdfs"
    output_dir = "data/augmented_images"
    
    if not os.listdir(input_dir):
        print(f"WAIT! Folder '{input_dir}' is empty. Move your images from Part 1 into this folder before running.")
    else:
        for filename in os.listdir(input_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                augment_image(os.path.join(input_dir, filename), output_dir)
        print("Success! Each image now has 5 augmented variants in data/augmented_images.")