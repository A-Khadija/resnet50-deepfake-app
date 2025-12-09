import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
from huggingface_hub import hf_hub_download
import timm  # pip install timm

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Deepfake Inspector", layout="wide")

# --- REPO, FILENAME, AND IMAGE SIZE CONFIGURATION ---
MODELS = {
    # --- RESNET (224x224) ---
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
    
    # --- EFFICIENTNET B4 (380x380) ---
    "EfficientNet B4 (TL)": {
        "repo_id": "obm-ml/Efficientnetb4-TL",
        "filename": "efficientnetb4_best.pkl",  
        "img_size": 380
    },
    "EfficientNet B4 (FN)": {
        "repo_id": "obm-ml/Efficientnetb4-FN",
        "filename": "efficientnetb4_finetuned_best.pkl",  
        "img_size": 380
    },

    # --- XCEPTION (299x299) ---
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
    """
    Recreates the EXACT architecture used during training.
    """
    if "ResNet50" in model_key:
        # Standard ResNet50 with modified FC
        model = models.resnet50(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        return model
        
    elif "EfficientNet" in model_key:
        # Standard EfficientNet B4 with modified Classifier
        model = models.efficientnet_b4(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, 2)
        return model
        
    elif "Xception" in model_key:
        # REPLICATE YOUR TRAINING LOGIC EXACTLY
        model = timm.create_model("xception", pretrained=False, num_classes=2)
        
        # We must replicate the 'head replacement' logic so keys match (fc.1.weight vs fc.weight)
        if hasattr(model, "fc"):
            in_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(in_features, 2)
            )
        elif hasattr(model, "classifier"):
            in_features = model.classifier.in_features
            model.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(in_features, 2)
            )
        elif hasattr(model, "head"):
            in_features = model.head.in_features
            model.head = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(in_features, 2)
            )
        return model
        
    return None

@st.cache_resource
def load_model(model_key):
    config = MODELS[model_key]
    try:
        # 1. Download
        model_path = hf_hub_download(repo_id=config["repo_id"], filename=config["filename"])
        
        # 2. Load Checkpoint
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # 3. Handle State Dict (Weights Only) vs Full Model
        if isinstance(checkpoint, dict):
            # It's a dictionary of weights -> Build architecture first
            model = build_model_architecture(model_key)
            if model is None:
                return "Architecture not defined"
            
            # Clean keys if they were saved with DataParallel ('module.' prefix)
            new_state_dict = {}
            for k, v in checkpoint.items():
                name = k.replace("module.", "") if k.startswith("module.") else k
                new_state_dict[name] = v
                
            # Load weights
            model.load_state_dict(new_state_dict, strict=False)
            
        else:
            # It's a full model object (Older save format)
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
        # ResNet
        if hasattr(model, 'layer4'):
            return model.layer4[-1]
        
        # EfficientNet
        if hasattr(model, 'features'):
            return model.features[-1]
            
        # Xception (Timm)
        # Timm Xception usually puts the last conv in .act4 or .conv4 depending on version
        # We try to grab the last sequential block or specific layer
        if "Xception" in model_name:
            if hasattr(model, 'act4'): return model.act4
            if hasattr(model, 'conv4'): return model.conv4
            # Fallback: traverse children to find last conv
            layers = list(model.children())
            for layer in reversed(layers):
                if isinstance(layer, nn.Conv2d):
                    return layer
        
        return None
    except:
        return None

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # We ONLY register the forward hook here.
        # The backward gradient capture is now handled dynamically inside the forward pass.
        self.target_layer.register_forward_hook(self.save_activation)

    def save_activation(self, module, input, output):
        self.activations = output
        
        # --- CRITICAL FIX ---
        # Instead of a module-level backward hook (which crashes on in-place ops),
        # we register a hook on the tensor itself.
        if output.requires_grad:
            output.register_hook(self.save_gradient)

    def save_gradient(self, grad):
        # Tensor hooks receive just the gradient tensor
        self.gradients = grad

    def __call__(self, x, class_idx=None):
        # 1. Reset state
        self.gradients = None
        self.activations = None
        
        # 2. Forward Pass
        output = self.model(x)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
            
        # 3. Backward Pass
        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()
        
        # 4. Generate CAM
        if self.gradients is None or self.activations is None:
            return None # Safety check if hooks didn't fire
            
        gradients = self.gradients[0]
        activations = self.activations[0]
        
        # Global Average Pooling
        weights = torch.mean(gradients, dim=(1, 2))
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=device)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        # ReLU and Normalization
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

# Sidebar
st.sidebar.header("Configuration")
options = ["All Models"] + list(MODELS.keys())
selected_option = st.sidebar.selectbox("Select Model Mode", options)

uploaded_file = st.file_uploader("Upload Image for Analysis", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Original Input", width=300)

    if st.button("Start Analysis"):
        with st.spinner("Processing..."):
            
            if selected_option == "All Models":
                models_to_run = list(MODELS.keys())
                st.info(f"Running Ensemble Analysis on {len(models_to_run)} models...")
            else:
                models_to_run = [selected_option]

            results_accumulator = []
            
            if len(models_to_run) > 1:
                cols = st.columns(3)
            else:
                cols = [st.container()]

            # --- MAIN LOOP ---
            for idx, model_name in enumerate(models_to_run):
                current_col = cols[idx % 3] if len(models_to_run) > 1 else cols[0]
                
                with current_col:
                    st.divider()
                    st.write(f"**{model_name}**")
                    
                    # 1. Load Model
                    model_or_error = load_model(model_name)
                    if isinstance(model_or_error, str):
                        st.error(f"Failed to load: {model_or_error}")
                        continue
                    
                    model = model_or_error
                    
                    # 2. Process Image (Unique Size per Model)
                    req_size = MODELS[model_name]["img_size"]
                    img_tensor = process_image(image, req_size).to(device)

                    try:
                        # 3. Inference
                        target_layer = get_target_layer(model, model_name)
                        
                        with torch.no_grad():
                            output = model(img_tensor)
                            probs = F.softmax(output, dim=1)
                            conf, pred = torch.max(probs, 1)
                            label = CLASS_NAMES[pred.item()]
                            confidence_val = conf.item()

                        results_accumulator.append({
                            "model": model_name,
                            "label": label,
                            "confidence": confidence_val
                        })

                        # 4. Display Result
                        st.caption(f"Input Size: {req_size}x{req_size}")
                        if label == "Fake":
                            st.error(f"FAKE ({confidence_val:.2%})")
                        else:
                            st.success(f"REAL ({confidence_val:.2%})")

                        # 5. Grad-CAM
                        if target_layer:
                            cam_extractor = GradCAM(model, target_layer)
                            with torch.enable_grad():
                                activation_map = cam_extractor(img_tensor, class_idx=pred.item())
                                overlay, heatmap = visualize_cam(activation_map, image)
                                st.image(overlay, caption=f"CAM ({model_name})", use_container_width=True)
                        else:
                            st.warning("Layer hook failed (No CAM)")
                        
                    except Exception as e:
                        st.error(f"Inference Error: {e}")

            # --- CONSENSUS LOGIC ---
            if selected_option == "All Models" and results_accumulator:
                st.divider()
                st.header("🏆 Final Consensus Verdict")
                
                fake_votes = [r for r in results_accumulator if r['label'] == 'Fake']
                real_votes = [r for r in results_accumulator if r['label'] == 'Real']
                
                n_fake = len(fake_votes)
                n_real = len(real_votes)
                total = len(results_accumulator)
                
                if n_fake > n_real:
                    final_verdict = "FAKE"
                    avg_conf = sum([r['confidence'] for r in fake_votes]) / n_fake if n_fake > 0 else 0
                    color_func = st.error
                elif n_real > n_fake:
                    final_verdict = "REAL"
                    avg_conf = sum([r['confidence'] for r in real_votes]) / n_real if n_real > 0 else 0
                    color_func = st.success
                else:
                    final_verdict = "UNCERTAIN"
                    avg_conf = 0.0
                    color_func = st.warning

                c1, c2 = st.columns([2, 1])
                with c1:
                    color_func(f"### Majority: {final_verdict}")
                    st.write(f"**Votes:** {n_fake} Fake vs {n_real} Real")
                    if total > 0:
                        st.progress(n_fake / total, text="Fake Vote Share")
                
                with c2:
                    st.metric("Avg Confidence (Winner)", f"{avg_conf:.1%}")
