import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from huggingface_hub import hf_hub_download  

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Deepfake Inspector", layout="wide")

# HUGGING FACE REPO DETAILS
REPO_ID = "KhadijaAsehnoune12/resnet50-deepfake-models"

# Filenames
MODEL_FILES = {
    "ResNet50 (Transfer Learning)": "deepfake_model_transferlearning.pkl",
    "ResNet50 (Fine-Tuning)": "deepfake_model_finetuning.pkl"
}

CLASS_NAMES = ['Real', 'Fake']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. MODEL LOADING (Via Hugging Face) ---
@st.cache_resource
def load_model(model_name):
    filename = MODEL_FILES[model_name]
    try:
        with st.spinner(f"Downloading {model_name} from Hugging Face..."):
            model_path = hf_hub_download(repo_id=REPO_ID, filename=filename)
        
        # Load the model
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Handle DataParallel (if the model was trained on multiple GPUs)
        if isinstance(checkpoint, torch.nn.DataParallel):
            model = checkpoint.module
        else:
            model = checkpoint
            
        model.to(device)
        model.eval()
        
        for param in model.parameters():
            param.requires_grad = True
            
        return model
    except Exception as e:
        st.error(f"Error loading {model_name}: {e}")
        return None

# --- 3. PREPROCESSING & GRAD-CAM ---
def process_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        output = self.model(x)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()
        
        gradients = self.gradients[0]
        activations = self.activations[0]
        weights = torch.mean(gradients, dim=(1, 2))
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=device)
        
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-7)
        return cam.cpu().detach().numpy()

def visualize_cam(mask, img_pil):
    heatmap = cv2.resize(mask, (img_pil.width, img_pil.height))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    original = np.array(img_pil)
    superimposed = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)
    return superimposed, heatmap

# --- 4. STREAMLIT APP UI ---
st.title("Deepfake Detection with Grad-CAM")
st.markdown("Use this tool to switch between models and **visualize** what the AI is looking at.")

# Sidebar
st.sidebar.header("Settings")
selected_model_name = st.sidebar.radio("Select Model Architecture", list(MODEL_FILES.keys()))

# Load Model
model = load_model(selected_model_name)

if model:
    st.sidebar.success(f"Loaded: {selected_model_name}")

    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        col1, col2 = st.columns([1, 1])
        image = Image.open(uploaded_file).convert('RGB')
        with col1:
            st.image(image, caption="Original Image", use_column_width=True)

        if st.button("Analyze & Explain"):
            with st.spinner("Running Inference..."):
                img_tensor = process_image(image).to(device)
                
                try:
                    # Attempt to find the last convolutional layer (usually layer4 for ResNet)
                    # We check if it exists first
                    target_layer = None
                    if hasattr(model, 'layer4'):
                         target_layer = model.layer4[-1]
                    
                    if target_layer is None:
                        st.error("Could not find 'layer4' in the model. Grad-CAM cannot run.")
                    else:
                        cam_extractor = GradCAM(model, target_layer)
                        activation_map = cam_extractor(img_tensor)
                        
                        # Prediction
                        with torch.no_grad():
                            output = model(img_tensor)
                            probs = F.softmax(output, dim=1)
                            conf, pred = torch.max(probs, 1)
                            label = CLASS_NAMES[pred.item()]
                            
                        # Display Result
                        st.divider()
                        if label == "Fake":
                            st.error(f"Prediction: **FAKE** (Confidence: {conf.item()*100:.2f}%)")
                        else:
                            st.success(f"Prediction: **REAL** (Confidence: {conf.item()*100:.2f}%)")
                            
                        overlay, heatmap = visualize_cam(activation_map, image)
                        
                        st.subheader("Visual Explanation (Grad-CAM)")
                        c1, c2 = st.columns(2)
                        with c1:
                            st.image(heatmap, caption="Heatmap (Where it looked)", use_column_width=True)
                        with c2:
                            st.image(overlay, caption="Overlay", use_column_width=True)

                except AttributeError as e:
                    st.error(f"Could not hook into model layer. Error: {e}")
