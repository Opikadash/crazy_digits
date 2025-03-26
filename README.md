# crazy_digits
This repository contains the handwritten written digits recognition using CNN and MNIST.

# Handwritten Digit Recognition (MNIST + User Corrections)

This project is a real-time handwritten digit recognition system using **PyTorch, Flask, and HTML/CSS/JS**. It allows users to draw digits, get predictions, and submit corrections if the model is wrong. The model **automatically retrains on user corrections** to improve over time! 

---

## Features
- Draw a digit on the canvas and get a prediction
- Submit corrections if the model is wrong
- Automatic model retraining on MNIST + user-corrected data
- Predict again using the updated model
- Flask backend with PyTorch deep learning
- Interactive UI with HTML, CSS, and JavaScript

*(Add a GIF or Screenshot of your app here)*

---

## Project Structure
```
📁 Handwritten-Digit-Recognition
│── 📁 corrections/       # Stores user-corrected images
│── 📁 templates/         # HTML files for UI
│── 📄 app.py             # Flask backend (API for prediction & correction)
│── 📄 trainer.py         # Model training & retraining script
│── 📄 mnist_cnn.pt       # Trained PyTorch model (saved weights)
│── 📄 requirements.txt   # Required Python libraries
│── 📄 README.md          # Project documentation
```

---

## Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Opikadash/crazy_digits.git
cd crazy_digits
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Flask App
```bash
python app.py
```
🔹 Now open: **(http://127.0.0.1:5000/)** in your browser.

### 4️⃣ Train the Model (First-time setup)
If running for the first time, train the model using:
```bash
python trainer.py
```
This will train the model on the MNIST dataset and save it as **mnist_cnn.pt**.

---

## How to Use

### Step 1: Draw a Digit
- Open the app in your browser.
- Draw a digit on the canvas.

### Step 2: Get a Prediction
- Click the **"Predict"** button.
- The model will predict the drawn digit.

### Step 3: Submit Corrections (if wrong)
- If the model predicts incorrectly, enter the correct digit label.
- Click **"Submit Correction"**.
- The model will automatically retrain.

### Step 4: Predict Again
- Click **"Predict Again"**.
- The model will now use the updated dataset!

---

## Technologies Used
- **Python** (Flask, PyTorch, NumPy)
- **TorchVision** (for MNIST dataset)
- **HTML, CSS, JavaScript** (for UI)
- **Canvas API** (for drawing digits)
- **Base64 Image Processing** (to send images to the server)
- **Auto-Retraining Pipeline** (trains model on corrections)

---

## Future Improvements
- Improve the UI with a better drawing experience
- Use a pretrained CNN for better accuracy
- Add deployment support (**Heroku, AWS, or Streamlit**)

---

## License
This project is **open-source** and available under the **MIT License**.

