
# Deep Learning for Pediatric Pneumonia Detection and Classification in Chest X-Rays

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)](https://github.com/AliAoun/Deep-Learning-Pediatric-Pneumonia-Detection-Classification.git)

**A transfer learning-based deep neural network for automated classification of chest X-rays to detect pneumonia in pediatric patients**

[Overview](#overview) • [Features](#features) • [Installation](#installation) • [Dataset](#dataset) • [Usage](#usage) • [Results](#results) • [Contributing](#contributing)

</div>

---

## 📋 Overview

This project implements a state-of-the-art deep learning solution for the automated detection and classification of pneumonia in pediatric chest X-rays. Using **transfer learning with DenseNet-161**, the model achieves high accuracy in distinguishing between healthy lungs and those affected by pneumonia, with the potential for real-world clinical deployment.

### 🎯 Key Objectives

- ✅ Develop an accurate and efficient CNN for pneumonia classification
- ✅ Implement transfer learning to leverage pre-trained features
- ✅ Handle imbalanced dataset through class weighting
- ✅ Achieve clinical-grade accuracy for reliable diagnosis assistance
- ✅ Provide a reproducible, production-ready pipeline

---

## ⚙️ Features

### Core Technologies

| Component | Technology |
|-----------|-----------|
| **Framework** | PyTorch |
| **Pre-trained Model** | DenseNet-161 (ImageNet weights) |
| **Optimizer** | Adam with Learning Rate Scheduling |
| **Loss Function** | Cross-Entropy Loss with class weighting |
| **GPU Support** | CUDA-enabled training |

### Technical Highlights

- 🧠 **Transfer Learning**: Leverages DenseNet-161 pre-trained on ImageNet
- ⚖️ **Class Weighting**: Handles imbalanced dataset automatically
- 📊 **Data Augmentation**: Multiple augmentation techniques including:
  - Random horizontal flips
  - Random rotation (±10°)
  - Random grayscale conversion
  - Random affine transformations
- 📈 **Learning Rate Scheduling**: StepLR scheduler for optimal convergence
- 💾 **Model Checkpointing**: Saves best model based on validation loss

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA 11.0+ (for GPU acceleration)
- pip or conda package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/AliAoun/Deep-Learning-Pediatric-Pneumonia-Detection-Classification.git
   cd pneumonia-detection-dl
   ```

2. **Create a virtual environment**
   ```bash
   # Using conda
   conda create -n pneumonia-env python=3.10
   conda activate pneumonia-env
   
   # Or using venv
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Requirements

```
torch>=2.0.0
torchvision>=0.15.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.12.0
numpy>=1.21.0
pandas>=1.3.0
jupyter>=1.0.0
```

---

## 📊 Dataset

### Source

The project uses the **Pediatric Chest X-ray Pneumonia dataset** from Kaggle:

🔗 [Dataset Link](https://www.kaggle.com/datasets/andrewmvd/pediatric-pneumonia-chest-xray)

### Dataset Structure

```
Pediatric Chest X-ray Pneumonia/
├── train/
│   ├── NORMAL/        (1,341 images)
│   └── PNEUMONIA/     (3,875 images)
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

### Dataset Characteristics

- **Total Images**: ~5,856 pediatric chest X-rays
- **Image Format**: JPEG, grayscale
- **Image Size**: Variable (resized to 256×256, center-cropped to 224×224)
- **Class Distribution**: Imbalanced (NORMAL: 1,341 | PNEUMONIA: 3,875)
- **Ages**: Pediatric patients (age 1-5 years)

### Class Imbalance Handling

The dataset suffers from class imbalance (~3:1 ratio). This is addressed through:

- **Class Weighting**: Automatically computed weights inversely proportional to class frequency
- **Formula**: $\text{weight}_{\text{class}} = 1 - \frac{\text{samples}_{\text{class}}}{\text{total}_{\text{samples}}}$

---

## 🚀 Usage

### Quick Start

1. **Prepare your dataset**
   ```bash
   # Download from Kaggle and extract to project directory
   cd data/
   unzip archive.zip
   ```

2. **Run the training pipeline**
   ```bash
   jupyter notebook pneumonia_detection_DL_classification.ipynb
   ```

### Training Configuration

Key hyperparameters in the notebook:

```python
# Model
model = models.densenet161(pretrained=True)
epochs = 15
batch_size = 64

# Optimizer
learning_rate = 0.001
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

# Scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

# Loss
loss_function = nn.CrossEntropyLoss()
```

### Training Pipeline Steps

1. **Data Import & Augmentation**
   - Load images from train/NORMAL and train/PNEUMONIA folders
   - Apply transformations (resize, crop, augment)
   - Split into 70% train, 30% validation

2. **Model Definition**
   - Load pre-trained DenseNet-161
   - Freeze feature extractor weights
   - Replace classifier with custom 2-class output layer

3. **Training**
   - Train for 15 epochs
   - Monitor training and validation loss/accuracy
   - Save best model based on validation loss

4. **Evaluation**
   - Test on held-out test set
   - Generate classification report
   - Compute accuracy and other metrics

### Making Predictions

```python
# Load trained model
model.load_state_dict(torch.load('best-model-weighted.pt'))
model.eval()

# Get predictions on test data
images, labels, probs, preds, accuracy = get_probs_and_preds(model, test_loader)

# Print results
print(f'Test Accuracy: {accuracy:.4f}')
print(classification_report(labels, preds))
```

---

## 📈 Results

### Model Performance Metrics (Test Set)

The DenseNet-161 model was evaluated on a held-out test set containing **624 pediatric chest X-ray images**.  
The following metrics were computed from the final trained model.

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **89.74%** |
| **Precision (Pneumonia)** | **88%** |
| **Recall (Pneumonia)** | **97%** |
| **F1-Score (Pneumonia)** | **92%** |
| **Precision (Normal)** | **95%** |
| **Recall (Normal)** | **77%** |
| **F1-Score (Normal)** | **85%** |


### Training Curves

The model exhibits:
- ✅ **Smooth convergence** over 15 epochs
- ✅ **Best validation loss ≈ 0.137**, indicating stable optimization
- ✅ **No severe overfitting**, with close alignment between training and validation curves
- ✅ Consistent improvement in accuracy across epochs


### Confusion Matrix

The confusion matrix reveals:
- ✔ **High true positive rate for pneumonia detection** (recall = 97%)
- ✔ **Low false negative rate**, which is critical for clinical screening
- ✔ Some misclassification of NORMAL cases, reflecting an intentional bias toward detecting pneumonia
- ✔ Overall balanced performance despite dataset class imbalance

---

## 🔧 Model Architecture

### DenseNet-161 Overview

```
DenseNet-161
├── Feature Extraction (frozen)
│   ├── Conv2d (3 → 96)
│   ├── DenseBlock-1 through DenseBlock-4
│   ├── Transition layers
│   └── BatchNorm → ReLU
│
└── Classifier (trainable)
    └── Linear (2208 → 2)  [NORMAL, PNEUMONIA]
```

### Transfer Learning Strategy

- **Freeze**: All feature extraction layers from pre-trained ImageNet weights
- **Train**: Only the custom classifier layer
- **Rationale**: Leverages learned features while focusing on task-specific classification

### Optional: Fine-tuning

For improved performance, unfreeze all parameters after initial training:

```python
# Unfreeze all parameters
for param in model.parameters():
    param.requires_grad = True

# Re-train with lower learning rate (0.0001)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
```

---

## 📋 Project Structure

```
pneumonia-detection-dl/
├── pneumonia_detection_DL_classification.ipynb  # Main training notebook
├── data/
│   └── Pediatric Chest X-ray Pneumonia/
│       ├── train/
│       ├── val/
│       └── test/
├── models/
│   └── best-model-weighted.pt                   # Trained model weights
├── results/
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   └── classification_report.txt
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 🔍 Key Functions

### Training Step
```python
def training_step(model, loader, loss_function):
    """Execute one training epoch"""
    # Forward pass → Loss computation → Backpropagation → Weight update
```

### Evaluation Step
```python
def evaluate_model(model, loader, loss_function):
    """Evaluate model on validation/test data"""
    # Forward pass (no gradients) → Loss & Accuracy computation
```

### Accuracy Calculation
```python
def calculate_accuracy(outputs, labels):
    """Compute prediction accuracy"""
    _, predictions = torch.max(outputs, dim=1)
    accuracy = (predictions == labels).sum() / len(predictions)
```

---

## 🚀 Future Enhancements

- [ ] **Ensemble Methods**: Combine multiple models for improved robustness
- [ ] **Attention Mechanisms**: Add attention layers for interpretability
- [ ] **Grad-CAM Visualization**: Visualize model decision-making regions
- [ ] **Web Interface**: Deploy as Flask/FastAPI REST API
- [ ] **Mobile Deployment**: Convert to TensorFlow Lite for edge devices
- [ ] **Multi-class Extension**: Extend to classify multiple pneumonia types
- [ ] **Real-time Inference**: Optimize for hospital PACS integration
- [ ] **Data Privacy**: Implement federated learning for multi-center training

---

## ⚠️ Important Considerations

### Clinical Disclaimer

⚠️ **This model is for research and educational purposes only.**
- Not approved for clinical diagnosis without professional medical validation
- Should be used as a **decision support tool**, not a replacement for radiologists
- Always consult qualified healthcare professionals for medical decisions

### Dataset Imbalance

- Dataset has a 3:1 pneumonia to normal ratio
- Class weighting is essential to prevent bias
- Consider stratified k-fold cross-validation for robust evaluation

### Overfitting Prevention

- Monitor validation loss throughout training
- Early stopping based on best validation performance
- Use data augmentation to prevent memorization

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/your-feature`)
3. **Make** your changes and commit (`git commit -m 'Add your feature'`)
4. **Push** to the branch (`git push origin feature/your-feature`)
5. **Open** a Pull Request

### Contribution Guidelines

- Follow PEP 8 style guide for Python code
- Add comments for complex logic
- Include docstrings for functions
- Test changes before submitting PR
- Update README if adding new features

---

## 📝 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

The dataset is available under the [CC0 License](https://www.kaggle.com/datasets/andrewmvd/pediatric-pneumonia-chest-xray) from Kaggle.

---

## 📧 Contact & Support

- **Author**: Syed Muhammad Ali Aoun
- **Email**: syed.m.ali.aoun@gmail.com
- **GitHub Issues**: [Report bugs and request features](../../issues)
- **Discussion**: [Project discussions](../../discussions)

---

## 🙏 Acknowledgments

- **Dataset Provider**: [Andrew Ng and team](https://www.kaggle.com/andrewmvd/) - Kaggle Pediatric Chest X-ray Pneumonia dataset
- **Model Architecture**: PyTorch DenseNet-161 implementation
- **Inspiration**: ResNet, VGGNet, and modern CNN architectures
- **References**:
  - Huang et al. (2017) - Densely Connected Convolutional Networks
  - Krizhevsky et al. (2012) - ImageNet Classification with Deep CNNs
  - Medical imaging best practices and standards

---

## 📚 References

1. **DenseNet Paper**: Huang, G., Liu, Z., Van Der Maaten, L., & Weinberger, K. Q. (2017). Densely Connected Convolutional Networks. CVPR.
2. **Transfer Learning**: Yosinski, J., et al. (2014). How transferable are features in deep neural networks? NIPS.
3. **Medical Imaging**: Ronneberger, O., et al. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI.
4. **PyTorch Documentation**: https://pytorch.org/docs/stable/
5. **Class Imbalance Handling**: He, H., & Garcia, E. A. (2009). Learning from Imbalanced Data. IEEE TKDE.

---

<div align="center">

**If you found this project helpful, please consider giving it a ⭐ Star!**

Made with ❤️ for better pediatric healthcare

</div>
