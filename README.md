# 📘 Adversarial Machine Learning: FGSM & PGD Attacks on CIFAR-10

This project demonstrates how **adversarial attacks** can drastically degrade the performance of a Convolutional Neural Network (CNN) trained on the **CIFAR-10** dataset.
Two major white-box attacks are implemented:

* **Fast Gradient Sign Method (FGSM)**
* **Projected Gradient Descent (PGD)**

The project evaluates model robustness across different perturbation strengths, computes Attack Success Rates (ASR), and visualizes adversarial examples vs. clean images.

---

## Objective

This project studies how a convolutional neural network (CNN) trained on the CIFAR-10 dataset
can be intentionally misled by small, carefully constructed perturbations to the input images. The
chosen project direction is adversarial attack, using two gradient based methods: the Fast Gradient
Sign Method (FGSM) and Projected Gradient Descent (PGD). The main objective is to quantify
how the classifier’s accuracy and the attack success rate change under different perturbation budgets
and, for PGD, different numbers of attack steps. A further goal is to compare FGSM and PGD in
terms of their ability to reduce accuracy while keeping the perturbed images visually close to the
original samples, and to illustrate these effects through visual examples and perturbation heatmaps.

---

## 🚀 Project Structure
```
📁 Adversarial_Attack_Project/
│── adversarial_project.ipynb
│── README.md
│── requirements.txt
│── /data
│── /images   (optional – saved visualization outputs)
```

---

## 🧠 1. Overview

This project:

* Trains a custom **SimpleCNN** on CIFAR-10
* Achieves **83.92% clean test accuracy**
* Generates adversarial examples using FGSM and PGD
* Computes **Adversarial Accuracy** and **Attack Success Rate (ASR)**
* Evaluates attacks across multiple ε values and PGD iteration counts
* Includes visualizations such as:
  * Clean vs. FGSM vs. PGD
  * 10-image grids
  * Perturbation heatmaps
  * 3-row combined layouts

---

## 📦 2. Dataset

**CIFAR-10** contains:

* 60,000 images
* 32×32 RGB
* 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)

Training transformations:

* Random crop
* Random horizontal flip
* Standard CIFAR normalization

---

## 🏗️ 3. Model Architecture

The **SimpleCNN** consists of:

* Three convolution blocks: `Conv → ReLU → MaxPool`
* Flatten layer
* Two fully connected layers

This lightweight architecture keeps training fast while still being expressive enough for adversarial research.

---

## 🎯 4. Training Results

The model was trained for **40 epochs** with SGD + Momentum + Weight Decay.

### Final Clean Test Accuracy:
```
83.92%
```

This is the baseline accuracy before applying FGSM and PGD attacks.

---

## ⚡ 5. FGSM Attack

FGSM perturbs the input using:
```
x_adv = x + ε * sign(∇x Loss)
```

### FGSM Result (ε = 0.05):
```
Clean accuracy: 83.91%
Adversarial accuracy: 33.17%
ASR: 66.83%
```

### Effect of Epsilon (ε):

| ε    | Adv Accuracy | ASR    |
|------|--------------|--------|
| 0.00 | 83.89%       | 16.11% |
| 0.01 | 72.09%       | 27.91% |
| 0.03 | 49.00%       | 51.00% |
| 0.05 | 33.17%       | 66.83% |
| 0.10 | 12.14%       | 87.86% |

**Observation:**
Increasing ε rapidly reduces model accuracy and increases the attack success rate.

---

## 🔥 6. PGD Attack

PGD applies iterative FGSM-like steps with projection back into the ε-ball.

### PGD Result (ε = 0.05, 10 steps):
```
Adversarial accuracy: 18.75%
ASR: 81.25%
```

### Effect of PGD Steps (ε = 0.05):

| Steps | Adv Accuracy | ASR    |
|-------|--------------|--------|
| 5     | 23.22%       | 76.78% |
| 10    | 18.75%       | 81.25% |
| 20    | 17.70%       | 82.30% |
| 40    | 17.45%       | 82.55% |

**Observation:**
More steps produce stronger attacks, with diminishing returns after ~20 iterations.

---

## 🖼️ 7. Visualizations

The notebook includes:

* **Clean vs FGSM vs PGD** comparisons
* **10-image side-by-side grids**
* **Perturbation heatmaps**
* **3-row combined layouts** (Clean → FGSM → PGD)

**Key Insight:**
Adversarial images look almost identical to clean images, yet predictions change significantly — showing how vulnerable CNNs can be.

---

## 🔧 8. Requirements

Dependencies used in this project:
```
torch
torchvision
numpy
matplotlib
```

All dependencies are listed in `requirements.txt`.

---

## ▶️ 9. How to Run

### 1. Clone the repository
```bash
git clone https://github.com/BALaka-18/Adversarial-Attacks-on-Deep-Neural-Networks---FGSM-PGD.git
cd Adversarial-Attacks-on-Deep-Neural-Networks---FGSM-PGD
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the notebook
```bash
jupyter notebook adversarial_project.ipynb
```

GPU is recommended for faster PGD iteration performance.

---

MS Data Science – University of Alabama at Birmingham (UAB)
Course: **CS685 – Foundations of Data Science**

---

## 🏁 11. Summary

This project demonstrates:

* CNNs achieve strong performance on clean CIFAR-10 images
* FGSM and PGD perturbations drastically reduce accuracy even with minimal noise
* PGD outperforms FGSM at the same epsilon
* Larger ε or more PGD steps increase attack success
* Adversarial robustness remains a critical challenge in deep learning systems

---

## 📄 License

This project is open-source and available under the MIT License.

---

## 📧 Contact

For questions or collaboration, feel free to reach out to: officialbalaka@gmail.com
