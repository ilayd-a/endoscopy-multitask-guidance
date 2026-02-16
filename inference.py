import os
import torch
import random
import config
from monai.transforms import Compose, LoadImage, EnsureChannelFirst, ScaleIntensity, Resize, ToTensor
from torchvision import models
from config import DATA_DIR
from config import MODELS_DIR

# Uses apple gpu or nvidia if available or fallback to cpu :)
device = config.get_device()

#
dataset_path = DATA_DIR/"EndoscopicBladderTissue"
class_folders = ["HGC", "LGC", "NST", "NTL"]

# Load your trained model
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

#random images from each class
print("Selecting random images from each class...")
image_paths = []
for class_name in class_folders:
    class_dir = os.path.join(dataset_path, class_name)
    if os.path.exists(class_dir):
        images = [f for f in os.listdir(class_dir) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        if images:
            # Pick 2 random images per class
            random.shuffle(images)
            for img_name in images[:2]:
                img_path = os.path.join(class_dir, img_name)
                image_paths.append(img_path)
            print(f"  {class_name}: 2 random images")
        else:
            print(f"  WARNING: No images in {class_name}")
    else:
        print(f"  ERROR: Folder not found: {class_dir}")

print(f"\nTesting {len(image_paths)} random images...")

# image processing 
batch = []
for path in image_paths:
    try:
        img = transform(path)
        batch.append(img)
    except Exception as e:
        print(f"Error loading {path}: {e}")

if batch:
    batch = torch.stack(batch).to(device)
    
    with torch.no_grad():
        outputs = model(batch)
        preds = torch.argmax(outputs, dim=1)
        # Get confidence scores
        probs = torch.nn.functional.softmax(outputs, dim=1)
    
    print("\n" + "="*60)
    print("INFERENCE RESULTS")
    print("="*60)
    
    correct = 0
    total = 0
    
    for path, pred, prob in zip(image_paths, preds, probs):
        filename = os.path.basename(path)
        true_class = os.path.basename(os.path.dirname(path))
        predicted_class = class_folders[pred.item()]
        confidence = prob[pred.item()].item()
        
        is_correct = (predicted_class == true_class)
        if is_correct:
            correct += 1
        total += 1
        
        print(f"\n {filename}")
        print(f"   True: {true_class}")
        print(f"   Pred: {predicted_class} ({confidence:.2%})")
        print(f"   Status: {'CORRECT' if is_correct else 'WRONG'}")
        
        # Show all class probabilities
        for i, cls in enumerate(class_folders):
            print(f"     {cls}: {prob[i]:.2%}")
    
    print("\n" + "="*60)
    print(f"📊 ACCURACY: {correct}/{total} correct ({correct/total:.2%})")
    print("="*60)
else:
    print("No images could be loaded!")