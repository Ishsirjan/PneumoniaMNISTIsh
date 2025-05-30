
 PneumoniaMNIST – ResNet-50 Transfer Learning Project
 ---

📌 Problem Statement

This project involves fine-tuning a deep learning model (ResNet-50) on the **PneumoniaMNIST** dataset to detect pneumonia from grayscale chest X-ray images. The task is part of Phase-1 assessment for the position of Project Research Scientist-I in an ICMR-funded project.

---

📁 Project Structure

```
.
├── train.ipynb              # Full training, evaluation, and fine-tuning code
├── requirements.txt         # All required Python packages
├── README.md                # Project summary and instructions
└── slides.pdf               # Summary presentation (3 slides)
```

---

🧪 Dataset

- PneumoniaMNIST (MedMNIST v2)  
- Shape: `(N, 28, 28)` grayscale images  
- Class 0: Normal (minority), Class 1: Pneumonia (majority)

---

🏗️ Model Architecture

- Base Model: ResNet-50 (pretrained on ImageNet)
- Input shape: 224×224×3 (converted from 28×28×1)
- Layers:
  - Resize → Grayscale to RGB
  - ResNet50 (frozen initially)
  - GlobalAveragePooling
  - Dense(128, ReLU) + Dropout(0.5)
  - Dense(1, Sigmoid via logits)

---

🧠 Training Strategy

| Phase         | Notes                                    |
|---------------|------------------------------------------|
|   Phase 1     | Freeze ResNet-50, train top layers       |
|   Phase 2     | Unfreeze top 50 layers, fine-tune model  |

- Batch Size: 8 (memory-efficient)
- Loss Function: Focal Loss (γ=2.0, α=0.25) to address class imbalance
- Optimizers: Adam + ReduceLROnPlateau + EarlyStopping
- Augmentation: Horizontal flip

---
⚖️ Class Imbalance Handling

- Dataset is heavily skewed (Class 0 ≪ Class 1)
- Used **oversampling for minority class
- Focal loss reduces focus on well-classified majority samples
- Metrics used emphasize minority class (precision, recall)

---

📊 Evaluation Metrics

- Accuracy: 82.2%
- Precision: 87.2%
- Recall: 83.9%
- F1 Score (Class 1): 86%
- ROC AUC Score: 0.91
- Confusion matrix and classification report included

---

 ▶️ How to Run

1. Clone the repo
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the training notebook:
   ```bash
   jupyter notebook train.ipynb
   ```

---

## 📦 Requirements

```txt
tensorflow>=2.12
numpy
scikit-learn
matplotlib
```

---
  Author

**Ishsirjan Kaur Chandok**  
Contact available on request.
