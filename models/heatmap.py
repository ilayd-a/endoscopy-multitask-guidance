import os
import random
import torch
import numpy as np
import cv2
from monai.transforms import Compose, LoadImage, EnsureChannelFirst, ScaleIntensity, Resize, ToTensor
from torchvision import models
import torch.nn.functional as F

from config import MODELS_DIR, DATA_DIR, get_device

# MPS apple silicon
device = get_device()

class_folders = ["HGC", "LGC", "NST", "NTL"]

# Load trained model
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_folders))
# Change the path to your model checkpoint
model.load_state_dict(torch.load(MODELS_DIR/"bladder_model.pth", map_location=device))
model = model.to(device)
model.eval()

transform = Compose([
    LoadImage(image_only=True),
    EnsureChannelFirst(),
    ScaleIntensity(),
    Resize((224, 224)),
    ToTensor()
])

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1)
        loss = output[:, class_idx]
        loss.backward()
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1)
        cam = F.relu(cam).squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam

gradcam = GradCAM(model, model.layer4)

#DATASET PATH
dataset_path = DATA_DIR/"EndoscopicBladderTissue"

#Find first image from each class
image_paths = []
print("Finding test images...")
for class_name in class_folders:
    class_dir = os.path.join(dataset_path, class_name)
    if os.path.exists(class_dir):
        # Get all image files
        images = [f for f in os.listdir(class_dir) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        if images:
            random.shuffle(images)
            # Take first image
            img_path = os.path.join(class_dir, images[0])
            image_paths.append(img_path)
            print(f"  {class_name}: {images[0]}")
        else:
            print(f"  WARNING: No images found in {class_name}")
    else:
        print(f"  ERROR: Folder not found: {class_dir}")

print(f"\nGenerating heatmaps for {len(image_paths)} images...")

# Create output directory
output_dir = "heatmap_results"
os.makedirs(output_dir, exist_ok=True)

for img_path in image_paths:
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        continue
    
    try:
        # Load and transform image
        img_tensor = transform(img_path).unsqueeze(0).to(device)
        
        # Get prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            pred = torch.argmax(outputs, dim=1).item()
            confidence = torch.nn.functional.softmax(outputs, dim=1)[0][pred].item()
        
        print(f"\n{os.path.basename(img_path)}")
        print(f"  True class: {os.path.basename(os.path.dirname(img_path))}")
        print(f"  Predicted: {class_folders[pred]} ({confidence:.2%})")
        
        # Generate heatmap
        cam = gradcam.generate(img_tensor, pred)
        cam_resized = cv2.resize(cam, (224, 224))
        
        # Load original image
        original = cv2.imread(img_path)
        if original is None:
            print(f"  ERROR: Could not read image with OpenCV")
            continue
            
        original = cv2.resize(original, (224, 224))
        
        # Create heatmap overlay
        heatmap = cv2.applyColorMap(np.uint8(cam_resized * 255), cv2.COLORMAP_JET)
        overlay = (0.4 * heatmap + 0.6 * original).astype(np.uint8)
        
        # Save
        filename = os.path.basename(img_path)
        save_path = os.path.join(output_dir, f"CAM_{filename}")
        cv2.imwrite(save_path, overlay)
        print(f"  Saved: {save_path}")
        
        # Also save original and heatmap separately
        cv2.imwrite(os.path.join(output_dir, f"original_{filename}"), original)
        cv2.imwrite(os.path.join(output_dir, f"heatmap_{filename}"), heatmap)
        
    except Exception as e:
        print(f"  ERROR processing {img_path}: {e}")

print(f"\n Done! Heatmaps saved in '{output_dir}/' folder")