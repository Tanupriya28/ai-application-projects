
# 🧠 Explainable Deepfake Image Detection System

An end-to-end AI system for detecting AI-generated deepfake images using Transfer Learning with EfficientNet, enhanced with Explainable AI (Grad-CAM) and deployed as a real-time web application.

---

## 🚨 Problem Statement

Deepfake images generated using advanced AI techniques are increasingly being used for:

- Misinformation and fake news  
- Identity fraud and impersonation  
- Cybercrime and digital manipulation  

Human visual inspection is unreliable for detecting such manipulations.  
This project aims to build an automated, accurate, and explainable deepfake detection system.

---

## 🌍 Applications

- Social media content moderation  
- Digital forensic investigations  
- Fake identity prevention  
- News authenticity verification  
- Cybersecurity monitoring  

---

## 📁 Dataset

- Approximately **10,000 images**
- Two classes: **Real** and **Fake**
- Split into training and validation sets
- Images resized to **224×224**

### Preprocessing:
- Normalization  
- EfficientNet preprocessing  
- Data augmentation  

---

## 🧪 Baseline CNN Model

A custom Convolutional Neural Network was implemented as a baseline:

- 3 convolutional layers with pooling  
- Fully connected classifier  
- Sigmoid activation  

### Results:
- Training Accuracy: ~94%  
- Validation Accuracy: ~75%  
- Overfitting observed  

---

## 🚀 Transfer Learning with EfficientNet

- Used **EfficientNetB0 pretrained on ImageNet**  
- Leveraged pretrained visual features  
- Improved convergence and generalization  

---

## ⚙️ Data Augmentation

- Random horizontal flipping  
- Random rotation  
- Random zoom  
- Random contrast  

Benefits: reduced overfitting and better real-world robustness.

---

## ❄️ Freezing & Fine Tuning Strategy

### Step 1 – Freeze backbone  
- Trained only classifier layers  

### Step 2 – Fine tuning  
- Unfroze layers with small learning rate  

### Step 3 – Partial unfreezing  
- Fine-tuned deeper layers only  

---

## 📈 Final Performance

- Validation Accuracy: **~85–86%**  
- Stable learning curves  
- Improved generalization  

---

## 🔍 Explainable AI (Grad-CAM)

- Highlights manipulated facial regions  
- Improves model transparency  
- Supports forensic analysis  

---

## 🧠 Forensic Intelligence Layer

- Full-image + face-based predictions  
- Confidence estimation  
- Reliability scoring  
- Consistency analysis  
- Artifact indicators  

---

## 🌐 Web Application Features

- Batch upload support  
- Real-time inference  
- Heatmap visualization  
- Face detection  
- PDF report generation  
- Prediction history  

---

## 📂 Project Structure
deepfake-ai/
│
├── templates/
│ ├── base.html
│ ├── detect.html
│ ├── history.html
│
│── screenshots
│
├── static/
│ ├── style2.css
│ └── app.js
│
├── app.py
├── requirements.txt
└── README.md

---

## ⚡ Installation

### 1️⃣ Clone repository
git clone https://github.com/yourusername/deepfake-ai.git
cd deepfake-ai
### 2️⃣ Create virtual environment 
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
### 3️⃣ Install dependencies
pip install -r requirements.txt
________________________________________
## ▶️ Running the Application
python app.py
Then open:
http://127.0.0.1:5000/
________________________________________
## 🛠 Technologies Used
•	Python
•	TensorFlow / Keras
•	EfficientNet
•	OpenCV
•	Flask
•	NumPy
•	ReportLab
•	HTML, CSS, JavaScript
________________________________________
## 🔮 Future Improvements
•	Larger datasets
•	Video deepfake detection
•	Live webcam processing
•	Vision transformers
•	Cloud deployment
________________________________________
## Conclusion
This project delivers a complete explainable AI pipeline for deepfake detection, combining strong model performance, reasoning, and real-time deployment.
________________________________________
