import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
from huggingface_hub import hf_hub_download
import timm 
import gc  # <--- Added for memory management

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Deepfake Inspector", layout="wide")

# Reduce PyTorch memory footprint on CPU
torch.set_num_threads(4) 

# --- REPO, FILENAME, AND IMAGE SIZE CONFIGURATION ---
MODELS = {
    "ResNet50 (Transfer Learning)": {
        "repo_id": "KhadijaAsehnoune12/resnet50-deepfake-models",
        "filename": "deepfake_model_transferlearning.pkl",
        "img_size": 224
    },
    "ResNet50 (Fine-Tuning)": {
        "repo_id": "KhadijaAsehnoune12/resnet50-deepfake-models",
        "filename": "deepfake_model_finetuning.pkl",
        "img_size": 224
    },
    "EfficientNet B4 (Transfer Learning)": {
        "repo_id": "obm-ml/Efficientnetb4-TL",
        "filename": "efficientnetb4_best.pkl",  
        "img_size": 380
    },
    "EfficientNet B4 (Fine-Tuning)": {
        "repo_id": "obm-ml/Efficientnetb4-FN",
        "filename": "efficientnetb4_finetuned_best.pkl",  
        "img_size": 380
    },
    "Xception (Transfer Learning)": {
        "repo_id": "HoudaTag/xception_TransfertLearnin",
        "filename": "xception_transfertLearning.pkl", 
        "img_size": 299 
    },
    "Xception (Fine-Tuning)": {
        "repo_id": "HoudaTag/xception_TransfertLearnin",
        "filename": "xception_finetuned.pkl", 
        "img_size": 299 
    }
}

CLASS_NAMES = ['Real', 'Fake']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. MODEL ARCHITECTURE BUILDER ---
def build_model_architecture(model_key):
    if "ResNet50" in model_key:
        model = models.resnet50(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        return model
        
    elif "EfficientNet" in model_key:
        model = models.efficientnet_b4(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, 2)
        return model
        
    elif "Xception" in model_key:
        model = timm.create_model("legacy_xception", pretrained=False, num_classes=2)
        
        if hasattr(model, "fc"):
            in_features = model.fc.in_features
            model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(in_features, 2))
        elif hasattr(model, "classifier"):
            in_features = model.classifier.in_features
            model.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(in_features, 2))
        elif hasattr(model, "head"):
            in_features = model.head.in_features
            model.head = nn.Sequential(nn.Dropout(0.5), nn.Linear(in_features, 2))
        return model
    return None
    
# --- CRITICAL FIX: max_entries=1 prevents hoarding all models in RAM ---
@st.cache_resource(max_entries=1, show_spinner=False)
def load_model_cached(model_key):
    # This wrapper function handles the heavy lifting
    return _load_model_logic(model_key)

def _load_model_logic(model_key):
    config = MODELS[model_key]
    try:
        model_path = hf_hub_download(repo_id=config["repo_id"], filename=config["filename"])
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        if isinstance(checkpoint, dict):
            model = build_model_architecture(model_key)
            if model is None: return "Architecture not defined"
            
            new_state_dict = {}
            for k, v in checkpoint.items():
                name = k.replace("module.", "") if k.startswith("module.") else k
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict, strict=False)
        else:
            if isinstance(checkpoint, torch.nn.DataParallel):
                model = checkpoint.module
            else:
                model = checkpoint
        
        model.to(device)
        model.eval() # Always default to eval to save memory
        return model

    except Exception as e:
        return f"Error: {str(e)}"

# --- 3. DYNAMIC PREPROCESSING ---
def process_image(image, size):
    transform = transforms.Compose([
        transforms.Resize((size, size)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# --- 4. GRAD-CAM HELPERS ---
def get_target_layer(model, model_name):
    try:
        if hasattr(model, 'layer4'): return model.layer4[-1] # ResNet
        if hasattr(model, 'features'): return model.features[-1] # EfficientNet
        if "Xception" in model_name:
            if hasattr(model, 'act4'): return model.act4
            return list(model.modules())[-2] # Fallback
        return list(model.modules())[-2]
    except:
        return None

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.handle = self.target_layer.register_forward_hook(self.save_activation)

    def save_activation(self, module, input, output):
        self.activations = output
        if output.requires_grad:
            output.register_hook(self.save_gradient)

    def save_gradient(self, grad):
        self.gradients = grad

    def close(self):
        # Clean up hooks to free memory
        self.handle.remove()

    def __call__(self, x, class_idx=None):
        self.gradients = None
        self.activations = None
        
        output = self.model(x)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
            
        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()
        
        if self.gradients is None or self.activations is None:
            return None
            
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

# --- 5. STREAMLIT APP UI ---
st.title("Deepfake Detection System")
st.markdown("Multi-Model Consensus: **ResNet (224)**, **EfficientNet (380)**, **Xception (299)**")

options = ["All Models"] + list(MODELS.keys())
selected_option = st.sidebar.selectbox("Select Model Mode", options)
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Original Input", width=300)

    if st.button("Start Analysis"):
        with st.spinner("Initializing..."):
            
            # Decide which models to run
            if selected_option == "All Models":
                models_to_run = list(MODELS.keys())
                st.info(f"Running sequential analysis on {len(models_to_run)} models. Please wait...")
            else:
                models_to_run = [selected_option]

            results_accumulator = []
            
            # Setup columns
            if len(models_to_run) > 1:
                cols = st.columns(3)
            else:
                cols = [st.container()]

            # --- MAIN LOOP ---
            progress_bar = st.progress(0)
            
            for idx, model_name in enumerate(models_to_run):
                current_col = cols[idx % 3] if len(models_to_run) > 1 else cols[0]
                
                # Update progress
                progress_bar.progress((idx + 1) / len(models_to_run))
                
                with current_col:
                    st.divider()
                    st.write(f"**{model_name}**")
                    
                    # 1. Load Model (Uses cached loader with max_entries=1)
                    model_or_error = load_model_cached(model_name)
                    
                    if isinstance(model_or_error, str):
                        st.error(f"Load Failed")
                        continue
                    
                    model = model_or_error
                    
                    # 2. Process Image
                    req_size = MODELS[model_name]["img_size"]
                    img_tensor = process_image(image, req_size).to(device)

                    try:
                        # 3. Inference
                        target_layer = get_target_layer(model, model_name)
                        
                        # Enable gradients ONLY for CAM calculation
                        with torch.set_grad_enabled(True):
                            output = model(img_tensor)
                            probs = F.softmax(output, dim=1)
                            conf, pred = torch.max(probs, 1)
                            label = CLASS_NAMES[pred.item()]
                            confidence_val = conf.item()

                            # 4. Grad-CAM
                            if target_layer:
                                cam_extractor = GradCAM(model, target_layer)
                                activation_map = cam_extractor(img_tensor, class_idx=pred.item())
                                overlay, heatmap = visualize_cam(activation_map, image)
                                st.image(overlay, caption=f"CAM ({model_name})", use_column_width=True)
                                cam_extractor.close() # Important: Remove hooks
                            else:
                                st.warning("No CAM available")

                        # Display Text Result
                        if label == "Fake":
                            st.error(f"FAKE ({confidence_val:.2%})")
                        else:
                            st.success(f"REAL ({confidence_val:.2%})")

                        results_accumulator.append({
                            "model": model_name, "label": label, "confidence": confidence_val
                        })

                    except Exception as e:
                        st.error(f"Error: {e}")
                    
                    # --- CRITICAL: FORCE MEMORY CLEANUP ---
                    # Delete local references
                    del img_tensor
                    del output
                    # Force Garbage Collection
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            progress_bar.empty()

            # --- CONSENSUS LOGIC ---
            if selected_option == "All Models" and results_accumulator:
                st.divider()
                st.header("Final Consensus Verdict")
                
                fake_votes = [r for r in results_accumulator if r['label'] == 'Fake']
                real_votes = [r for r in results_accumulator if r['label'] == 'Real']
                
                n_fake = len(fake_votes)
                n_real = len(real_votes)
                
                if n_fake > n_real:
                    final_verdict = "FAKE"
                    avg_conf = sum([r['confidence'] for r in fake_votes]) / n_fake
                    st.error(f"### Majority: {final_verdict} ({n_fake} vs {n_real})")
                elif n_real > n_fake:
                    final_verdict = "REAL"
                    avg_conf = sum([r['confidence'] for r in real_votes]) / n_real
                    st.success(f"### Majority: {final_verdict} ({n_real} vs {n_fake})")
                else:
                    st.warning("### Result: UNCERTAIN (Tie)")
