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
import gc
import time

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Deepfake Inspector", layout="wide")

# Force CPU usage to avoid CUDA memory overhead on free tier
# (Unless you have a paid GPU instance, CPU is safer for memory)
device = torch.device("cpu")
torch.set_num_threads(2) # Limit threads to reduce overhead

# --- REPO CONFIGURATION ---
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

# --- 2. MODEL BUILDER ---
def build_model_architecture(model_key):
    if "ResNet50" in model_key:
        model = models.resnet50(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
    elif "EfficientNet" in model_key:
        model = models.efficientnet_b4(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, 2)
    elif "Xception" in model_key:
        model = timm.create_model("legacy_xception", pretrained=False, num_classes=2)
        if hasattr(model, "fc"):
            model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(model.fc.in_features, 2))
        elif hasattr(model, "classifier"):
            model.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(model.classifier.in_features, 2))
        elif hasattr(model, "head"):
            model.head = nn.Sequential(nn.Dropout(0.5), nn.Linear(model.head.in_features, 2))
    else:
        return None
    return model

# --- NO CACHE (Low Memory Mode) ---
def load_model_uncached(model_key):
    config = MODELS[model_key]
    try:
        model_path = hf_hub_download(repo_id=config["repo_id"], filename=config["filename"])
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        if isinstance(checkpoint, dict):
            model = build_model_architecture(model_key)
            if model is None: return "Architecture error"
            
            clean_state = {k.replace("module.", ""): v for k, v in checkpoint.items()}
            model.load_state_dict(clean_state, strict=False)
        else:
            model = checkpoint.module if isinstance(checkpoint, torch.nn.DataParallel) else checkpoint
        
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        return str(e)

# --- 3. PREPROCESSING ---
def process_image(image, size):
    transform = transforms.Compose([
        transforms.Resize((size, size)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# --- 4. GRAD-CAM ---
def get_target_layer(model, model_name):
    try:
        if hasattr(model, 'layer4'): return model.layer4[-1]
        if hasattr(model, 'features'): return model.features[-1]
        if "Xception" in model_name:
            if hasattr(model, 'act4'): return model.act4
            return list(model.modules())[-2]
        return list(model.modules())[-2]
    except:
        return None

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        self.handle = target_layer.register_forward_hook(self.save_activation)

    def save_activation(self, module, input, output):
        self.activations = output
        if output.requires_grad:
            output.register_hook(self.save_gradient)

    def save_gradient(self, grad):
        self.gradients = grad

    def close(self):
        self.handle.remove()

    def __call__(self, x, class_idx=None):
        output = self.model(x)
        if class_idx is None: class_idx = output.argmax(dim=1).item()
        
        self.model.zero_grad()
        output[0, class_idx].backward()
        
        if self.gradients is None or self.activations is None: return None
        
        weights = torch.mean(self.gradients[0], dim=(1, 2))
        cam = torch.sum(weights[:, None, None] * self.activations[0], dim=0)
        cam = F.relu(cam)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-7)
        return cam.detach().cpu().numpy()

def visualize_cam(mask, img_pil):
    heatmap = cv2.resize(mask, (img_pil.width, img_pil.height))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(np.array(img_pil), 0.6, heatmap, 0.4, 0)

# --- 5. APP UI ---
st.title("Deepfake Detection System")

options = ["All Models"] + list(MODELS.keys())
selected = st.sidebar.selectbox("Mode", options)
upload = st.file_uploader("Image", type=["jpg", "png", "jpeg"])

if upload and st.button("Start Analysis"):
    image = Image.open(upload).convert('RGB')
    st.image(image, width=300, caption="Input")
    
    models_to_run = list(MODELS.keys()) if selected == "All Models" else [selected]
    results = []
    
    if len(models_to_run) > 1:
        cols = st.columns(3) # Creates 3 columns side-by-side
    else:
        cols = [st.container()]

    progress = st.progress(0)
    
    for i, name in enumerate(models_to_run):
        progress.progress((i + 1) / len(models_to_run))
        
        # --- ADD THIS TO SELECT THE CURRENT COLUMN ---
        current_col = cols[i % 3] if len(models_to_run) > 1 else cols[0]
        
        # --- CHANGE "with container:" TO "with current_col:" ---
        with current_col:
            st.divider()
            st.write(f"**{name}**")
            # 1. LOAD (No Cache)
            model = load_model_uncached(name)
            if isinstance(model, str):
                st.error(f"Failed: {model}")
                continue
                
            # 2. PREPARE
            tensor = process_image(image, MODELS[name]["img_size"]).to(device)
            
            # 3. INFERENCE
            try:
                target = get_target_layer(model, name)
                
                # Force gradients for frozen models (Fixes "Heatmap unavailable")
                if target:
                    for p in target.parameters(): p.requires_grad = True
                
                with torch.set_grad_enabled(True):
                    # CAM
                    cam_runner = GradCAM(model, target) if target else None
                    
                    # Predict
                    out = model(tensor)
                    probs = F.softmax(out, dim=1)
                    conf, pred = torch.max(probs, 1)
                    lbl = CLASS_NAMES[pred.item()]
                    val = conf.item()
                    
                    # Visualize
                    if cam_runner:
                        map_data = cam_runner(tensor, pred.item())
                        cam_runner.close() # Clean hook immediately
                        
                        if map_data is not None:
                            # Fixes "OpenCV Error"
                            viz = visualize_cam(map_data, image)
                            st.image(viz, caption=f"CAM: {lbl}", width=300)
                        else:
                            st.warning("Heatmap skipped (gradient missing)")
                    
                    if lbl == "Fake": st.error(f"FAKE ({val:.1%})")
                    else: st.success(f"REAL ({val:.1%})")
                    
                    results.append({"label": lbl, "confidence": val})
            
            except Exception as e:
                st.error(f"Error: {e}")
            
            # 4. NUCLEAR CLEANUP (Crucial for Streamlit Cloud)
            del model
            del tensor
            del out
            if 'cam_runner' in locals() and cam_runner: del cam_runner
            
            # Force Python to release memory NOW
            gc.collect()
            time.sleep(0.5) # Allow OS to reclaim RAM
            
    # --- RESTORED CONSENSUS LOGIC ---
    if selected == "All Models" and results_accumulator:
        st.divider()
        st.header("Final Consensus Verdict")

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
