import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION & FILE PATHS ---
st.set_page_config(page_title="Deepfake Inspector", layout="wide")

# UPDATE THIS DICTIONARY WITH YOUR EXACT FILENAMES
MODEL_FILES = {
    "Transfer Learning ResNet50": "deepfake_model_complete.pkl",
    "Fine-Tuning ResNet50": "deepfake_model_finetuning.pkl"
}

CLASS_NAMES = ['Real', 'Fake'] 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. GRAD-CAM CLASS ---
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Hook to capture gradients and activations
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        # Forward pass
        output = self.model(x)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Backward pass
        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()

        # Generate Heatmap
        gradients = self.gradients[0]
        activations = self.activations[0]
        
        # Global Average Pooling on gradients (weights)
        weights = torch.mean(gradients, dim=(1, 2))
        
        # Weighted combination of activations
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=device)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        cam = F.relu(cam) # ReLU to focus only on positive contributions
        
        # Normalize 0-1
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-7)
        
        return cam.cpu().detach().numpy()

# --- 3. UTILITIES ---
@st.cache_resource
def load_model(path):
    try:
        model = torch.load(path, map_location=device)
        model.eval()
        return model
    except Exception as e:
        return None

def process_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def visualize_cam(mask, img_pil):
    """Overlays the heatmap on the original image"""
    heatmap = cv2.resize(mask, (img_pil.width, img_pil.height))
    
    # Convert heatmap to RGB using colormap
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Superimpose
    original = np.array(img_pil)
    superimposed = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)
    return superimposed, heatmap

# --- 4. STREAMLIT APP ---
st.title(" Deepfake Detection with Grad-CAM")
st.markdown("Use this tool to switch between models and **visualize** what the AI is looking at.")

# Sidebar Controls
st.sidebar.header("Settings")
selected_model_name = st.sidebar.radio("Select Model Architecture", list(MODEL_FILES.keys()))
selected_model_path = MODEL_FILES[selected_model_name]

# Load Model
model = load_model(selected_model_path)

if model is None:
    st.error(f"Could not load **{selected_model_path}**. Make sure the file exists in this folder.")
else:
    st.sidebar.success(f"Loaded: {selected_model_name}")

    # --- MAIN INTERFACE ---
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        col1, col2 = st.columns([1, 1])
        
        image = Image.open(uploaded_file).convert('RGB')
        
        with col1:
            st.image(image, caption="Original Image", use_column_width=True)

        if st.button("Analyze & Explain"):
            with st.spinner("Running Inference & Grad-CAM..."):
                img_tensor = process_image(image).to(device)
                
                # 1. Initialize Grad-CAM
                # target_layer is usually 'layer4' for ResNet50
                try:
                    target_layer = model.layer4[-1] 
                    cam_extractor = GradCAM(model, target_layer)
                    
                    # 2. Forward & Backward Pass (via GradCAM class)
                    activation_map = cam_extractor(img_tensor)
                    
                    # 3. Get Prediction Results from the model (standard pass)
                    # We need a clean forward pass for probabilities
                    with torch.no_grad():
                        output = model(img_tensor)
                        probs = F.softmax(output, dim=1)
                        conf, pred = torch.max(probs, 1)
                        label = CLASS_NAMES[pred.item()]
                        
                    # 4. Display Prediction
                    st.divider()
                    if label == "Fake":
                        st.error(f"Prediction: **FAKE** (Confidence: {conf.item()*100:.2f}%)")
                    else:
                        st.success(f"Prediction: **REAL** (Confidence: {conf.item()*100:.2f}%)")
                        
                    # 5. Display Grad-CAM
                    overlay, heatmap = visualize_cam(activation_map, image)
                    
                    st.subheader("Visual Explanation (Grad-CAM)")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.image(heatmap, caption="Heatmap (Attention)", use_column_width=True)
                    with c2:
                        st.image(overlay, caption="Overlay (Where it looked)", use_column_width=True)
                        
                    st.info("The **Red/Warm** areas indicate the regions that contributed most to the decision.")

                except AttributeError:
                    st.error("Could not find `layer4` in your model. Please check your model architecture.")
