# CIFAR-100 Image Classification Project 🧠

This repository demonstrates image classification on the CIFAR-100 dataset using two different approaches:

1. **Custom CNN from Scratch**
   - Data Augmentation
   - Learning Rate Scheduler
   - Early Stopping

2. **Transfer Learning using ResNet-50**

## 📂 Project Structure
```
cifar100-classification/
├── scratch_model.ipynb              # CNN from scratch
├── resnet50_transfer_learning.ipynb # ResNet-50 transfer learning
├── train_cnn.py                     # Python script version of CNN (optional)
├── requirements.txt                 # Dependencies
├── README.md                        # Project documentation
└── .gitignore                       # Ignore unnecessary files
```

## 📊 Dataset
- **CIFAR-100**: 60,000 images (32×32), 100 classes
- 50,000 training images, 10,000 test images
- Automatically downloaded via `tf.keras.datasets`

## 🚀 Setup & Installation
Clone the repo and install dependencies:

```bash
git clone <your-repo-url>.git
cd cifar100-classification
pip install -r requirements.txt
```

## ▶️ Usage
Start Jupyter and run the notebooks:

```bash
jupyter notebook
```

- Run `scratch_model.ipynb` → CNN from scratch
- Run `resnet50_transfer_learning.ipynb` → ResNet-50 transfer learning

Or use the script version:

```bash
python train_cnn.py
```

## 📈 Features
- ✅ Data Augmentation
- ✅ Learning Rate Scheduler
- ✅ Early Stopping
- ✅ Top-5 Accuracy
- ✅ Transfer Learning with ResNet-50
- ✅ Model Checkpointing

## 🎉 Results
| Model              | Top-1 Accuracy | Top-5 Accuracy |
|--------------------|----------------|----------------|
| CNN (Scratch + DA) | ~65%           | ~89%           |
| ResNet-50 Transfer | ~65–75%        | ~85–90%        |

## 🧰 Requirements
See `requirements.txt` for dependencies.

---
Happy Learning & Training! 🚀
