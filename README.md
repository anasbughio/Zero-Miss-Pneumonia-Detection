# 🫁 Zero-Miss Pneumonia Detection System

**A Safety-First Ensemble Deep Learning Framework for Detecting Pneumonia in Chest X-Rays.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow%2FKeras-orange)](https://www.tensorflow.org/)
[![Status](https://img.shields.io/badge/Status-Completed-success)]()

## 📌 Project Overview (Project Kya Hai)
Medical diagnosis mein "Accuracy" se zyada zaroori "Sensitivity" (Recall) hoti hai. Agar AI model kisi bimar patient ko "Healthy" bata de (False Negative), toh ye patient ki jaan ke liye khatra ho sakta hai.

Is project ka main maqsad **False Negatives ko Zero karna** hai. Humne ek "Safety-First" approach use ki hai jisme:
1.  **Image Enhancement:** CLAHE use karke X-ray ko saaf kiya.
2.  **Ensemble Learning:** DenseNet-121 aur EfficientNet-B0 models ko combine kiya.
3.  **Golden Thresholding:** Prediction decision boundary ko optimize kiya ($t=0.23$) taake koi bhi pneumonia case miss na ho.

---

## 🚀 Key Features (Khasiyat)
* **Zero Misses:** Achieved **100% Sensitivity (Recall)** on the test set.
* **Preprocessing:** Used **CLAHE (Contrast Limited Adaptive Histogram Equalization)** to enhance lung opacities (dhundla-pan).
* **Hybrid Architecture:** Combines global features (DenseNet) and local textures (EfficientNet).
* **Class Imbalance Handling:** Used **Weighted Sparse Categorical Cross-Entropy** loss function.

---

## 📊 Methodology (Kaam Kaise Kiya)

### 1. Preprocessing (CLAHE)
Normal X-rays aksar low contrast hote hain. Humne **CLAHE** apply kiya taake viral pneumonia ke subtle signs (ground-glass opacities) highlight ho jayen.
* *Clip Limit:* 2.0
* *Tile Grid Size:* (8, 8)

### 2. Model Architecture
Humne "Heavy-Light" Ensemble strategy use ki:
* **Model 1: DenseNet-121** (Heavy) - Ye deep features aur patterns ko pakadta hai (Feature Reuse).
* **Model 2: EfficientNet-B0** (Light) - Ye fine textures aur edges ko detect karta hai.
* **Ensemble Strategy:** Weighted Average Voting.
    $$Final\_Score = 0.7 \times P_{DenseNet} + 0.3 \times P_{EfficientNet}$$

### 3. Golden Threshold ($t=0.23$)
Standard AI 0.5 (50%) par faisla karta hai. Humne threshold ko **0.23** par set kiya.
* *Logic:* Agar model ko 23% bhi shaq hai ke patient ko pneumonia hai, toh hum usay "Pneumonia" declare karte hain taake risk na liya jaye.

---

## 📈 Results (Natijay)

| Metric | Score | Meaning |
| :--- | :--- | :--- |
| **Recall (Sensitivity)** | **100%** | **Ek bhi bimar bacha miss nahi hua (Main Goal).** |
| **Accuracy** | 96.8% | Overall performance bohot high hai. |
| **Precision** | 97.4% | False Alarms bohot kam hain. |
| **F1-Score** | 0.98 | Perfect balance between Precision & Recall. |

> **Note:** Confusion Matrix shows **0 False Negatives** out of 390 positive test cases.

---

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** TensorFlow, Keras, NumPy, Pandas, Matplotlib, Seaborn, OpenCV.
* **Platform:** Google Colab (T4 GPU).

---

## 📂 Dataset
Used the **Kermany et al. Chest X-Ray Images (Pneumonia)** dataset.
* **Training:** 5,216 images
* **Test:** 624 images
* **Classes:** Normal vs. Pneumonia (Viral/Bacterial)

---

## 🔧 How to Run

1.  **Clone the Repo:**
    ```bash
    git clone [https://github.com/YourUsername/Zero-Miss-Pneumonia-Detection.git](https://github.com/YourUsername/Zero-Miss-Pneumonia-Detection.git)
    ```
2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the Notebook:**
    Open `Pneumonia_Detection_Final.ipynb` in Jupyter Notebook or Google Colab.

---

## 🤝 Contributing
Agar aap is project ko improve karna chahte hain (e.g., Try ResNet or Vision Transformers), feel free to fork and submit a Pull Request!

## 📜 License
This project is open-source under the **MIT License**.
