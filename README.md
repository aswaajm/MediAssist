# 🩺 MediAssist: Medical Report Generator

*A Transformer-based deep learning system for generating radiology reports from chest X-ray images.*

MediAssist uses a hybrid architecture combining **ResNet-50**, **spatial attention**, and a **Transformer decoder** to generate descriptive clinical text.  

The design is inspired by **Tienet**, combining convolutional and attention-based reasoning.

---

## 📌 Table of Contents

- [Model Architecture](#-model-architecture)
- [Key Features](#-key-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Configuration](#️-configuration)
- [Usage Workflow](#️-usage-complete-workflow)
- [Sample Results](#-sample-results)

---

## 🧠 Model Architecture

### 🔹 1. Image Encoder — ResNet-50

- Pretrained **ResNet-50**, final FC removed  
- Input: 224×224 image  
- Output: `batch × 2048 × 7 × 7` features  
- 1×1 conv reduces channels → `d_model = 512`

---

### 🔹 2. Spatial Attention

A simple convolution-based attention layer produces a **global saliency map**, highlighting important regions before decoding.

---

### 🔹 3. Transformer Decoder

- 2 decoder layers  
- Masked self-attention on text  
- Cross-attention with image features  
- Linear classification head → vocabulary logits  

---

## ✨ Key Features

✔ Centralized configuration (`config.py`)  
✔ Differential learning rates  
✔ Mixed Precision (AMP) for faster training  
✔ ResNet freezing for initial epochs  
✔ Beam Search decoding (`BEAM_WIDTH = 5`)  
✔ BLEU/ROUGE evaluation  
✔ Visualization tools 

---

## 📂 Project Structure

```text
.
├── config.py                          # Global settings
├── mediassist_model.py                # Model architecture
├── mediassist_dataset.py              # Dataset + preprocessing
├── mediassist_train.py                # Train model
├── mediassist_evaluate.py             # Evaluate on test set
├── mediassist_calculate_metrics.py    # BLEU/ROUGE
├── mediassist_visualize.py            # Curves
├── mediassist_predict.py              # Single-image prediction
├── train_split.csv
├── valid_split.csv
├── model_test.csv
├── checkpoints/
│   ├── tienet_report_transformer_best.pth
│   └── training_history.json
├── results/
│   ├── mediassist_test_predictions.csv
│   └── metrics.json
└── visualizations/
    ├── training_curves.png
    └── metrics.png
```

---

## 🔧 Installation

1. **Clone this repository** and navigate into the directory.

2. **Activate your virtual environment:**
   ```bash
   # Windows
   myenv\Scripts\activate
   
   # macOS/Linux
   source myenv/bin/activate
   ```

3. **Install all required dependencies:**
   ```bash
   pip install torch torchvision transformers pandas tqdm scikit-learn nltk rouge_score seaborn opencv-python matplotlib
   ```

4. **Download the NLTK punkt tokenizer** for metric calculation:
   ```python
   import nltk
   nltk.download('punkt')
   ```

---

## ⚙️ Configuration

All settings are controlled by `config.py`.

Before running any script, review the settings in this file. This file controls:

- File paths (`TRAIN_CSV_FILE`, `BEST_MODEL_PATH`, etc.)
- Model hyperparameters (`D_MODEL`, `N_HEAD`, `NUM_DECODER_LAYERS`)
- Training settings (`BATCH_SIZE`, `LEARNING_RATE`, `NUM_EPOCHS`)
- Image settings (`IMAGE_SIZE`)

This allows you to change a setting in one place and have it apply to all scripts.

---

## 🚀 Usage: Complete Workflow

Follow these steps in order to train, evaluate, and visualize the model.

### Step 1: Prepare Data & Config

- Download the MIMIC-CXR dataset from physionet.org
- Place your `train_split.csv`, `valid_split.csv`, and `model_test.csv` files in the root directory. Ensure they have the columns `image_path` and `report_text`.
- Open `config.py` and verify all paths and hyperparameters (like `BATCH_SIZE` or `NUM_EPOCHS`) are correct.

### Step 2: Train the Model

Run the training script. This script will automatically use the settings from `config.py`.

```bash
python mediassist_train.py
```

- The script will print training and validation loss for each epoch.
- The best model is saved to `checkpoints/tienet_report_transformer_best.pth`.
- A `training_history.json` file is saved for visualization.

### Step 3: Evaluate on Test Set

Use the trained model to generate predictions for your test set.

```bash
python mediassist_evaluate.py
```

- This reads from `model_test.csv` (defined in config).
- It loads the `tienet_report_transformer_best.pth` (from config).
- It generates a report for every image and saves the output to `results/mediassist_test_predictions.csv` (from config).

### Step 4: Calculate Metrics

Calculate the BLEU and ROUGE scores from the predictions file.

```bash
python mediassist_calculate_metrics.py
```

- This reads the `mediassist_test_predictions.csv` you just created.
- It prints the scores to the console.
- It saves the final scores to `results/metrics.json`.

### Step 5: Visualize Results

Run the visualization script to generate all plots.

```bash
python mediassist_visualize.py
```

This script will:
- Generate `training_curves.png` from `training_history.json`.
- Generate `metrics.png` (bar charts) from `metrics.json`.

All images are saved to the `visualizations/` folder.

### Step 6: Predict a Single Image

To test a single image, use the predict script.

```bash
python mediassist_predict.py --image "path/to/your/image.jpg"
```

This loads the best model and generates a report on the fly.

---

## 📊 Results

```json
{
    "BLEU-1": 0.3101,
    "BLEU-2": 0.2156,
    "BLEU-3": 0.1566,
    "BLEU-4": 0.1174,
    "ROUGE-1": 0.3377,
    "ROUGE-2": 0.1404,
    "ROUGE-L": 0.2672
}
```

