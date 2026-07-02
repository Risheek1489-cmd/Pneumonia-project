import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# -----------------------
# CONFIG
# -----------------------
MODEL_PATH = "model_best.pth"
IMAGE_PATH = "test.jpg"   # <-- change this
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------
# LOAD MODEL
# -----------------------
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

class_names = checkpoint["class_names"]

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))

model.load_state_dict(checkpoint["model_state"])
model = model.to(DEVICE)
model.eval()

# -----------------------
# IMAGE TRANSFORM
# -----------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# -----------------------
# LOAD IMAGE
# -----------------------
img = Image.open(IMAGE_PATH).convert("RGB")
img = transform(img).unsqueeze(0).to(DEVICE)

# -----------------------
# PREDICT
# -----------------------
with torch.no_grad():
    output = model(img)
    probs = torch.softmax(output, dim=1)

    conf, pred = torch.max(probs, 1)

label = class_names[pred.item()]

print("Prediction:", label)
print("Confidence:", conf.item()*100)