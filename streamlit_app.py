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

# ---  REPO, FILENAME, AND IMAGE SIZE CONFIGURATION ---
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
    "Xception (TL)": {
        "repo_id": "HoudaTag/xception_TransfertLearnin",
        "filename": "xception_transfertLearning.pkl", 
        "img_size": 299 
    },
    "Xception (FT)": {
        "repo_id": "HoudaTag/xception_TransfertLearnin",
        "filename": "xception_finetuned.pkl", 
        "img_size": 299 
    }
}

CLASS_NAMES = ['Real', 'Fake']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. MODEL LOADING ---
@st.cache_resource
def load_model(model_key):
    config = MODELS[model_key]
    try:
        model_path = hf_hub_download(repo_id=config["repo_id"], filename=config["filename"])
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        if isinstance(checkpoint, torch.nn.DataParallel):
            model = checkpoint.module
        else:
            model = checkpoint
            
        model.to(device)
        model.eval()
        
        # Enable gradients for Grad-CAM
        for param in model.parameters():
            param.requires_grad = True
        return model
    except Exception as e:
        return None 

# --- 3. DYNAMIC PREPROCESSING ---
def process_image(image, size):
    """
    Resizes and normalizes the image based on the specific model's requirement.
    """
    transform = transforms.Compose([
        transforms.Resize((size, size)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# --- 4. GRAD-CAM HELPERS ---
def get_target_layer(model, model_name):
    """
    Attempts to find the last convolutional layer based on architecture name/structure.
    """
    try:
        # ResNet usually uses 'layer4'
        if hasattr(model, 'layer4'):
            return model.layer4[-1]
        
        # EfficientNet usually uses 'features' or 'blocks' (getting the last block)
        if hasattr(model, 'features'):
            return model.features[-1]
            
        # Common fallback for simple sequential models
        if hasattr(model, 'net') and hasattr(model.net, 'features'):
             return model.net.features[-1]

        return None
    except:
        return None

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

# --- 5. STREAMLIT APP UI ---
st.title("Deepfake Detection System")
st.markdown("Multi-Model Consensus: **ResNet (224)**, **EfficientNet (380)**, **Xception (229)**")

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
            
            # Determine models to run
            if selected_option == "All Models":
                models_to_run = list(MODELS.keys())
                st.info(f"Running Ensemble Analysis on {len(models_to_run)} models...")
            else:
                models_to_run = [selected_option]

            results_accumulator = []
            
            # Grid Layout
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
                    model = load_model(model_name)
                    if model is None:
                        st.error(f"Failed to load. Verify Filename in Code.")
                        continue

                    # 2. Process Image (Unique Size per Model)
                    req_size = MODELS[model_name]["img_size"]
                    # We process fresh for every model to ensure correct resizing
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
                                st.image(overlay, caption=f"CAM ({model_name})", use_column_width=True)
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
