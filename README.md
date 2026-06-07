# A Multi-Class Flower Recognition System Using Custom CNNs and TensorFlow

An end-to-end computer vision solution designed to classify 20 distinct categories of flowers from raw image data. Built on a custom Convolutional Neural Network (CNN) architecture using TensorFlow 2, the pipeline includes automated data augmentation, performance optimization routines and a lightweight web interface for real-time inference.

## 💠 Dataset Architecture & Pipeline
The model is trained on a structured dataset comprising 16,253 images evenly distributed across 20 botanical classes.

#### Data Preprocessing & Optimization
- **Data Splits:** 80% Training (13,003 images), 20% Validation (3,250 images).
- **Memory Management:** Leveraged tf.data API buffering techniques (.cache() and .prefetch(buffer_size=AUTOTUNE)) to decouple disk I/O operations from model training steps, optimizing GPU/CPU utilization.
- **On-the-Fly Data Augmentation:** Implemented an integrated sequential preprocessing pipeline to eliminate overfitting: 
    - Horizontal Flips
    - Random Rotations (up to 10%)
    - Random Zoom & Contrast Adjustments (up to 10%)
    - Random Brightness Adjustments

## 💠 Model Architecture
The custom CNN was built from scratch using the Keras Sequential API, balancing model complexity with high computational efficiency (117,492 total parameters).

| Layer (Type)              | Specification / Hyperparameters                                             | Purpose                                        |
|---------------------------|-----------------------------------------------------------------------------|------------------------------------------------|
| Data Augmentation         | Sequential Layers (Resizing 180x180)                                        | Generalization & Input Standardization         |
| Rescaling                 | 1./255                                                                      | Min-Max Feature Normalization                  |
| Convolutional Blocks (x4) | 16 ⇾ 32 ⇾ 64 ⇾ 128 Filters (3x3 kernel) | Hierarchical Feature Extraction                |
| Regularization            | BatchNormalization (per block) + Dropout (0.3)                              | Internal Covariate Shift & Overfitting Control |
| Dimensionality Reduction  | GlobalAveragePooling2D                                                      | Feature Map Flattening                         |
| Dense Network             | 128 Units (ReLU) ⇾ 20 Units (Softmax)                           | Non-linear Mapping & Class Probabilities       |


## 💠 Training Performance
The model was compiled using the Adam Optimizer (LR = 0.001) and optimized via a Learning Rate Scheduler (decaying by a factor of 0.1 every 10 epochs) alongside an EarlyStopping callback monitoring validation accuracy.
- Final Training Accuracy: 95.78% (Loss: 0.1297)
- Final Validation Accuracy: 94.15% (Loss: 0.2030)

## 💠 Tech Stack & Dependencies
- Deep Learning Framework: TensorFlow 2.x, Keras
- Extended Math & CV: TensorFlow Addons (TFA), NumPy, Pillow (PIL)
- Visualization: Matplotlib
- Deployment Web App: Streamlit

## 💠 Getting Started
1. Clone the Repository
```bash
git clone https://github.com/salman-sayeed/Flower-Recognition-CNN-Model-Using-TensorFlow-v2.git
cd Flower-Recognition-CNN-Model-Using-TensorFlow-v2 
```

2. Install Dependencies
```bash
pip install tensorflow tensorflow-addons numpy pillow matplotlib streamlit
```

3. Run Inference via Python

To run a quick prediction on a local image using the Python environment:
```bash
from tensorflow.keras.models import load_model
# Load the pre-trained weights
model = load_model('Flower_Recognition_Model_v2.h5')
# Execute classification function using your local target path
```

4. Launch the Web UI

Run the interactive Streamlit application to upload your own images and see real-time classification metrics:

```bash
streamlit run app.py
```
