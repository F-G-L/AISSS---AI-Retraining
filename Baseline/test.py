import os
import pandas as pd
import torch
from PIL import Image
import torchvision.transforms as T
from model import get_model
import requests
from io import BytesIO

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

def log_info(message):
    info_path = os.path.join(CNN_PROJECT, "info.txt")
    with open(info_path, "a", encoding="utf-8") as f:
        f.write(message + "\n")

ISIC_FOLDER = os.path.join(CNN_PROJECT, "isic2018", "ISIC2018_Task3_Training_Input")
GT_CSV = os.path.join(CNN_PROJECT, "isic2018", "ISIC2018_Task3_Training_GroundTruth.csv")
MODEL_PATH = os.path.join(CNN_PROJECT, "best_model.pth")

classes = ["MEL","NV","BCC","AKIEC","BKL","DF","VASC"]

transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model(num_classes=len(classes), pretrained=False).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

df = pd.read_csv(GT_CSV)
df["label"] = df[classes].idxmax(axis=1)
gt_map = dict(zip(df["image"], df["label"]))

def load_image(path_or_url):
    try:
        if path_or_url.startswith("http"):
            response = requests.get(path_or_url)
            img = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            img = Image.open(path_or_url).convert("RGB")
        return transform(img).unsqueeze(0)
    except Exception as e:
        print(f"⚠️ Error loading {path_or_url}: {e}")
        return None

def predict_batch(image_list):
    correct, total = 0, 0
    for path in image_list:
        name = os.path.splitext(os.path.basename(path))[0]
        print(f"\n🔍 Processing: {path}")
        img_tensor = load_image(path)
        if img_tensor is None:
            continue
        img_tensor = img_tensor.to(device)
        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs, 1)
            label = classes[predicted.item()]
        true_label = gt_map.get(name, "Unknown")
        result = "✅ Correct" if label == true_label else "❌ Wrong"
        if label == true_label:
            correct += 1
        total += 1
        print(f"Predicted: {label} | True: {true_label} | {result}")

    if total > 0:
        accuracy = correct / total * 100
        print(f"\n📊 Summary: {correct}/{total} correct ({accuracy:.2f}% accuracy)")
        log_info(f"Test attempt | correct={correct} total={total} accuracy={accuracy:.4f}%")
    else:
        log_info("Test attempt | No valid images tested")

def show_ground_truth_range(start, end):
    for i in range(start, end + 1):
        img_name = f"ISIC_{i:07d}"
        true_label = gt_map.get(img_name, None)
        if true_label:
            print(f"{img_name}: {true_label}")
        else:
            print(f"{img_name}: ⚠️ Not found in CSV")

if __name__ == "__main__":
    print('Enter up to 20 image paths or URLs (press Enter when done).')
    print('Commands:')
    print(' - "sofanthiel" → test a range (e.g. 0034313–0034413)')
    print(' - "leihtnafos" → list true answers for a range')

    image_list = []
    while len(image_list) < 20:
        inp = input(f"Image {len(image_list)+1}: ").strip().strip('"').strip("'")

        if inp.lower() == "sofanthiel":
            range_str = input("Enter range (e.g. 0034313-0034413): ").strip()
            try:
                start, end = map(int, range_str.split("-"))
                temp_list = []
                for i in range(start, end + 1):
                    img_name = f"ISIC_{i:07d}.jpg"
                    full_path = os.path.join(ISIC_FOLDER, img_name)
                    if os.path.exists(full_path):
                        temp_list.append(full_path)
                print(f"✅ Added {len(temp_list)} images from {start}–{end}")
                predict_batch(temp_list)
            except Exception as e:
                print(f"⚠️ Invalid range: {e}")
            continue

        elif inp.lower() == "leihtnafos":
            range_str = input("Enter range (e.g. 0034313-0034413): ").strip()
            try:
                start, end = map(int, range_str.split("-"))
                print(f"\n📘 Ground truth labels for {start}–{end}:")
                show_ground_truth_range(start, end)
            except Exception as e:
                print(f"⚠️ Invalid range: {e}")
            continue

        if inp == "":
            break
        image_list.append(inp)

    if len(image_list) == 0:
        print("No images entered.")
        log_info("Test attempt | No images entered")
    else:
        predict_batch(image_list)
