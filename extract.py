import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

# Load the lightweight AI model (MobileNetV2)
print("Loading AI model...")
model = models.mobilenet_v2(pretrained=True)
'''It is a placeholder or consider a empty pipe in which output is same  as input.
It is used because torch expect forward function to be non empty , otherwise it will throw error 
'''
model.classifier = torch.nn.Identity() # Remove the text labels, keep only the math
model.eval() # Set to evaluation mode (saves memory)

# Standardize the image (crop and resize to 224x224)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create a dummy image to test our engine
test_image_path = 'test_image.jpg'
Image.new('RGB', (500, 500), color='red').save(test_image_path)

# Convert the image to a vector
print("Extracting vector fingerprint...")
img = Image.open(test_image_path).convert('RGB')
img_tensor = preprocess(img).unsqueeze(0)

with torch.no_grad(): # Don't track gradients, saves RAM
    vector = model(img_tensor).squeeze().numpy()

# Save the output for C++ to read later
output_file = 'embeddings.bin'
vector.astype(np.float32).tofile(output_file)
print(f"Success! Saved a {len(vector)}-dimension vector to {output_file}")