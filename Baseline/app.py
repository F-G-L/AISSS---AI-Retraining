import os
import torch
import torchvision.transforms as T
from model import get_model
from PIL import Image
import gradio as gr

def find_cnn_project():
    cur = os.path.abspath(__file__)
    while True:
        cur = os.path.dirname(cur)
        if os.path.basename(cur) == "Cnn_Project":
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            raise FileNotFoundError("Could not find folder named 'Cnn_Project' above this file.")
        cur = parent

CNN_PROJECT = find_cnn_project()

IMG_SIZE = 224
classes = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model path (inside Cnn_Project)
MODEL_PATH = os.path.join(CNN_PROJECT, "best_model.pth")

model = get_model(num_classes=len(classes), pretrained=False)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

def predict(image):
    img = Image.fromarray(image).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

    return {classes[i]: float(probs[i]) for i in range(len(classes))}

iface = gr.Interface(
    fn=predict,
    inputs=gr.inputs.Image(type="numpy"),
    outputs=gr.outputs.Label(num_top_classes=3)
)

iface.launch()
