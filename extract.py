import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import glob
import os

print("Loading AI model...")
model = models.mobilenet_v2(pretrained=True)
model.classifier = torch.nn.Identity() 
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- UPDATE: Use recursive glob to find images in subfolders ---
# The '**' means "look in all subfolders"
# Note: We check for multiple extensions just in case some are .jpeg or .png
image_paths = []
for ext in ('*.jpg', '*.jpeg', '*.png'):
    image_paths.extend(glob.glob(f"dataset/**/{ext}", recursive=True))

all_vectors = []

print(f"Found {len(image_paths)} images. Extracting features...")

if len(image_paths) == 0:
    print("Error: No images found. Make sure your images are inside the 'dataset' folder.")
    exit()

# Save the order of images
with open("image_mapping.txt", "w") as f:
    for idx, path in enumerate(image_paths):
        try:
            img = Image.open(path).convert('RGB')
            img_tensor = preprocess(img).unsqueeze(0)
            with torch.no_grad():
                vector = model(img_tensor).squeeze().numpy()
            all_vectors.append(vector)
            f.write(path + "\n")
            
            # Print progress every 100 images
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{len(image_paths)} images...")
                
        except Exception as e:
            print(f"Error processing {path}: {e}")

# Save the massive matrix of all vectors
if all_vectors:
    np_vectors = np.array(all_vectors, dtype=np.float32)
    np_vectors.tofile("database.bin")
    print(f"Database built! Saved {len(all_vectors)} vectors to database.bin")