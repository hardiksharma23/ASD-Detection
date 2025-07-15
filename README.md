# ASD Detection Project
<img width="1918" height="869" alt="Screenshot 2025-03-31 024518" src="https://github.com/user-attachments/assets/11f54054-4bbf-408c-927b-495bf6d21845" />

<img width="530" height="637" alt="Screenshot 2025-03-28 071824" src="https://github.com/user-attachments/assets/f55c646c-1c1f-4c87-b7f2-dbcd916aa373" />

## Deployment
- **Backend:** Deployed on AWS EC2, accessible via AWS API Gateway
- **Frontend:** Deployed on AWS Amplify

---

## Overview
This project provides a machine learning pipeline and web API for detecting autism spectrum disorder (ASD) features from facial images. It includes:
- A backend (Flask API) for image analysis and prediction.
- Training scripts and utilities for building and evaluating models on facial features (eyes, nose, lips).
- Data organization and instructions for training and inference.

---

## Features
- **Image Analysis API:**
  - The Flask backend (`api.py`) exposes an `/analyze` endpoint. You can POST an image, and it returns a prediction (autistic/non-autistic) and an annotated result image (base64-encoded).
- **Model Training:**
  - Training scripts and utilities for binary classification (autistic vs. non-autistic) using images of eyes, nose, and lips.
  - Models are trained using PyTorch and can leverage NVIDIA’s Apex for mixed precision and distributed training.
- **Data Structure:**
  - Data is organized into folders for training, validation, and testing, with subfolders for each facial feature and class.

---

## Requirements
- Python 3.12.1 (recommended)
- PyTorch
- torchvision
- numpy
- scikit-learn
- flask
- flask-cors
- tqdm
- tensorboard
- scipy
- ml_collections
- NVIDIA GPU (for training with Apex/mixed precision)

Install all Python dependencies with:
```bash
pip install -r ASD-Backend/requirements.txt
```

You may also need to install NVIDIA Apex for mixed precision training:
```bash
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install
cd ..
```

---

## Setup & Usage

### 1. Run the API
```bash
cd ASD-Backend
python api.py
```
The API will be available at `http://localhost:5000/analyze`.

#### Example API Usage
Send a POST request with an image file:
```bash
curl -X POST -F "image=@path_to_image.jpg" http://localhost:5000/analyze
```
Response:
```json
{
  "output": "autistic" | "non-autistic",
  "result_image": "<base64-encoded annotated image>"
}
```

### 2. Train a Model
- Prepare your data in the `ASD-Backend/Data/Train` and `ASD-Backend/Data/Valid` folders, organized by feature and class.
- Edit `data_utils.py` to set the correct dataset paths.
- Example training command:
  ```bash
  python train.py --name Eyes/checkpoint_eyes --pretrained_weights ./pretrained_weights/ViT-B_16.npz --dataset train --model_type ViT-B_16 --fp16 --fp16_opt_level 02
  ```


---

## Notes
- **Large model files and datasets are not included in the repository** due to GitHub file size limits. You must download or provide these separately.
- The backend expects model checkpoints to be present in `ASD-Backend/model_checkpoint/`.
- For deployment, ensure your environment can access the required model files.
- For more details on training, see `ASD-Backend/train_steps.txt`.

---

## Frontend: ASD Web App

### Overview
The `asd-Frontend` directory contains a modern React web application that serves as the user interface for the ASD Detection system. It allows users to upload facial images, sends them to the backend API for analysis, and displays the prediction and annotated result image.

### Features
- Clean, responsive UI built with React and Tailwind CSS
- Image upload and preview
- Sends images to the backend Flask API (`/analyze` endpoint)
- Displays prediction result (autistic/non-autistic)
- Shows annotated result image returned by the backend
- Error handling and loading states

### Requirements
- Node.js (v18+ recommended)
- npm or yarn

### Setup & Usage

1. **Install dependencies:**
   ```bash
   cd asd-Frontend
   npm install
   # or
   yarn install
   ```

2. **Start the development server:**
   ```bash
   npm run dev
   # or
   yarn dev
   ```
   The app will be available at `http://localhost:5173` (default Vite port).

3. **Connect to Backend:**
   - By default, the frontend sends requests to `http://localhost:5000/analyze`.
   - Make sure the backend Flask API is running on port 5000, or update the API URL in `src/App.jsx` if needed.

### Build for Production
```bash
npm run build
# or
yarn build
```
The production-ready files will be in the `dist/` directory.

---


