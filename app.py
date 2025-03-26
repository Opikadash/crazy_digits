from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import numpy as np
import base64
import io
import os
import subprocess
import time

app = Flask(__name__)

# Ensure corrections directory exists
if not os.path.exists("corrections"):
    os.makedirs("corrections")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.6)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)

def load_model():
    if os.path.exists("mnist_cnn.pt"):
        model.load_state_dict(torch.load("mnist_cnn.pt", map_location=device))
        model.eval()
    else:
        print("No trained model found. Train first!")
        return None

load_model()

def preprocess_image(image):
    image = Image.open(io.BytesIO(base64.b64decode(image.split(',')[1])))
    image = ImageOps.grayscale(image)
    image = ImageOps.invert(image)
    image = image.resize((28, 28))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    return transform(image).unsqueeze(0).to(device)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    image_data = request.json.get("image")
    input_img = preprocess_image(image_data)

    with torch.no_grad():
        output = model(input_img)
        _, prediction = torch.max(output, 1)

    return jsonify({"prediction": prediction.item()})

@app.route("/store_correction", methods=["POST"])
def store_correction():
    data = request.json
    image = Image.open(io.BytesIO(base64.b64decode(data["image"].split(',')[1])))
    label = data["correct_label"]
    image.save(f"corrections/{label}_{np.random.randint(10000)}.png")

    # ✅ Start retraining immediately
    subprocess.run(["python", "trainer.py"], check=True)

    # ✅ Reload the updated model
    load_model()

    return jsonify({"status": "success", "message": "Correction stored. Model retrained!"})

if __name__ == "__main__":
    app.run(debug=True)
