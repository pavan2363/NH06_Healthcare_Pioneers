<<<<<<< HEAD
🩺 Kidney Stone Detection using Deep Learning
📌 Project for Neurax Hackathon
📖 Overview

Kidney stone disease is a common urological problem that requires accurate detection for proper treatment.
In this project, we built an AI-powered solution that can:
✅ Detect kidney stones in X-ray/CT scan images
✅ Show the stone location with bounding boxes
✅ Count multiple stones if present
✅ Estimate stone size and provide visual + textual report
We integrated the trained model into a Streamlit Web Application for easy usage in clinical or research environments.

🛠️ Tech Stack

Python 3.11

PyTorch + YOLOv8 (Ultralytics)

Streamlit (WebApp)

OpenCV + Pillow (Image handling)

Kaggle Dataset (Kidney stone images with bounding boxes)



📂 Folder Structure

KidneyStoneDetection/
│── data/                 # Dataset (train/valid/test)
│── runs/                 # YOLO training runs (weights, results)
│── app.py                # Streamlit web application
│── data.yaml             # Dataset config file
│── requirements.txt      # Dependencies
│── README.md             # Project Documentation


⚙️ Installation

1️⃣ Clone this repository
git clone https://github.com/yourusername/KidneyStoneDetection.git
cd KidneyStoneDetection

2️⃣ Create & activate environment
conda create -n kidney python=3.11 -y
conda activate kidney

3️⃣ Install dependencies

If you have NVIDIA GPU with CUDA:

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121


Then install rest:

pip install ultralytics streamlit opencv-python pillow



🏋️ Training the Model

Train YOLOv8 model:

yolo detect train data=data.yaml model=yolov8n.pt epochs=50 imgsz=640    (You can also train your model according to your Dataset)


Trained weights will be saved in:

runs/detect/trainX/weights/best.pt



🔍 Running Inference

To test on a single image:

yolo detect predict model=runs/detect/trainX/weights/best.pt source="sample.jpg"

🌐 Run the Web Application

streamlit run app.py

"Then open: http://localhost:8501"



Features in WebApp:

Upload an X-ray image and CT scan images

Model detects stones and highlights them

Shows count, size, and bounding box location


🎯 Results

Detects kidney stones with bounding boxes

Works on multiple stones in the same scan

Real-time visualization in web app

🚀 Future Improvements

Extend dataset with CT & Ultrasound images

Deploy on cloud (AWS/GCP) for remote access

Add automated stone size measurement in mm

👨‍💻 Team

Developed as part of Neurax Hackathon 🧠💡

=======
# NeuraX-hackathon_project
>>>>>>> b4f668c (Initial commit)
