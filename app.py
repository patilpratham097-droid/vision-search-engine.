import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import subprocess
import os

# 1. Page Config
st.set_page_config(page_title="AI Image Search", layout="wide")
st.title("Reverse Image Search Engine")
st.write("Upload an image to find visually similar images in the database.")

# 2. Cache the AI model so it doesn't reload on every upload
@st.cache_resource
def load_model():
    model = models.mobilenet_v2(pretrained=True)
    model.classifier = torch.nn.Identity()
    model.eval()
    return model

model = load_model()

preprocess = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 3. Load the filenames mapping
try:
    with open("image_mapping.txt", "r") as f:
        image_paths = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    st.error("Error: image_mapping.txt not found. Run extract.py first.")
    st.stop()

# 4. The UI: File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded query image
    query_image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(query_image, caption="Your Search Image")

    with col2:
        with st.spinner("Analyzing image and searching database..."):
            # A. Convert upload to vector
            img_tensor = preprocess(query_image).unsqueeze(0)
            with torch.no_grad():
                query_vector = model(img_tensor).squeeze().numpy()
            
            # B. Save for C++
            query_vector.astype(np.float32).tofile("query.bin")

            # C. Run C++ Engine
            cpp_executable = "./search_engine" if os.name != "nt" else "search_engine.exe"
            result = subprocess.run([cpp_executable], capture_output=True, text=True)

            if result.returncode != 0:
                st.error("Error running C++ engine. Did you compile it?")
                st.stop()

            # D. Parse output indices
            top_indices = [int(idx) for idx in result.stdout.strip().split("\n") if idx]

        # 5. Display the Results
        st.subheader("Top 5 Closest Matches:")
        result_cols = st.columns(5)
        
        for idx, col_idx in zip(top_indices, range(5)):
            match_image_path = image_paths[idx]
            with result_cols[col_idx]:
                st.image(Image.open(match_image_path), caption=f"Match {col_idx+1}")