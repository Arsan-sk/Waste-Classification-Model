# 📖 WasteVision — Complete Code Explanation
### Understanding `train.py` and `app.py` from Absolute Zero

> **Who is this for?** Someone who has never studied neural networks, machine learning, computer vision, or Streamlit. We explain every single concept from scratch — what a pixel is, how a computer "sees" an image, why we use pre-trained neural networks, and exactly what every line of code does and *why* it exists.

---

## 🧠 Part 0 — The Foundations: Neural Networks and Computer Vision

### 1. What is an Image to a Computer?
When you look at a plastic bottle, your brain instantly recognizes the shape, label, and texture. A computer cannot do this directly. To a computer, an image is just **a giant grid of numbers**.

Images are made of pixels. In color images, each pixel is represented by three numbers:
* **R (Red)**: Intensity from 0 to 255
* **G (Green)**: Intensity from 0 to 255
* **B (Blue)**: Intensity from 0 to 255

For example, a solid green pixel is `[0, 255, 0]`. A white pixel is `[255, 255, 255]`.
In our training configuration, we resize all images to **224 × 224 pixels**. With 3 channels (RGB), each image is represented as a **3D grid of numbers** (a tensor):
$$\text{Input Shape} = 224 \text{ rows} \times 224 \text{ columns} \times 3 \text{ color channels} = 150,528 \text{ individual numbers}$$
The job of our neural network is to look at these 150,528 numbers and determine: *"This is a Plastic container (98% confidence)."*

---

### 2. How Does a Convolutional Neural Network (CNN) Work?
Instead of reading all 150,528 numbers at once like a simple spreadsheet, a CNN processes the image through layered filters:

```
[Raw Image Pixels (224x224x3)]
               │
               ▼
[Convolutional Layers]   ──► Detects basic edges, boundaries, and lighting gradients
               │
               ▼
[Deep Feature Layers]    ──► Combines edges into complex patterns (caps, textures, text)
               │
               ▼
[Global Pooling Layer]   ──► Compresses spatial features into a 1D vector (1280 features)
               │
               ▼
[Dense Classification]   ──► Reasons about the features and outputs 8 probabilities (one per class)
```

---

### 3. What is Transfer Learning?
Training a neural network from scratch requires millions of images and massive supercomputers. Instead, we use **Transfer Learning**:
1. We take a world-class model (**EfficientNetB0**), which was already trained by Google on **1.4 million images** (1,000 different classes).
2. EfficientNetB0 has already learned how to extract high-quality features (edges, curves, textures) from images.
3. We "cut off" its original 1,000-class decision head and glue on our own custom **classification head** designed for our 8 waste classes.
4. This allows us to achieve high accuracy even with a small dataset (80 images per class).

---

### 4. What is Two-Phase Training?
To train the model effectively without corrupting the knowledge inside the pre-trained EfficientNetB0 backbone:
* **Phase 1 (Warm-up)**: We **freeze** the backbone layers and train *only* our custom classification head. This teaches the head to map the backbone's features to waste categories.
* **Phase 2 (Fine-Tuning)**: We **unfreeze** the last 50 layers of the backbone and train the whole model with a **very small learning rate**. This allows the backbone to adapt slightly to the specific shapes and textures of waste items.

---

## 📄 Part 1 — Understanding `train.py` (The Training Pipeline)

Let's break down the logic of the training script section by section.

### 1. Imports and System Initialization
```python
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import json
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
...
```
* **`os` & environment flag**: We set `TF_ENABLE_ONEDNN_OPTS = '0'` to prevent floating-point variations on CPU calculations, ensuring reproducible training outputs.
* **`numpy` & `random`**: Used to set initial random seeds (`SEED = 42`) so that shuffling, augmentation, and layer weight initializations behave identically if you run the script multiple times.
* **`tensorflow`**: The engine driving the model architecture, weight updates, and training loop.

---

### 2. Configuration Settings (Hyperparameters)
```python
IMG_SIZE         = 224          # Target dimensions for EfficientNetB0
BATCH_SIZE       = 16           # Process 16 images at a time (suits small datasets)
EPOCHS_HEAD      = 40           # Phase 1 training epochs
EPOCHS_FINE      = 50           # Phase 2 fine-tuning epochs
LR_HEAD          = 3e-4         # Learning rate for Phase 1
LR_FINE          = 2e-5         # 15x smaller learning rate for fine-tuning
FINE_TUNE_AT     = -50          # Number of backbone layers to unfreeze in Phase 2
LABEL_SMOOTHING  = 0.1          # Softens target classes to reduce overconfidence
```
* **`BATCH_SIZE = 16`**: Large batches crash small GPU/CPU memories. 16 is small enough to run anywhere and updates model weights frequently.
* **`LR_FINE = 2e-5`**: Fine-tuning learning rate is very small to avoid "catastrophic forgetting"—which happens if large updates destroy pre-trained weights.
* **`LABEL_SMOOTHING = 0.1`**: Instead of target vectors being $[0, 0, 1, 0]$ (100% confidence), they are softened to $[0.0125, 0.0125, 0.9125, 0.0125]$. This prevents the model from becoming overconfident and overfitting.

---

### 3. Data Generators & Real-Time Augmentation
```python
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # EfficientNet preprocessor
    validation_split=0.2,                     # 20% reserved for testing
    rotation_range=25,                        # Rotate up to +/- 25 degrees
    width_shift_range=0.2,                    # Horizontal translation
    height_shift_range=0.2,                   # Vertical translation
    shear_range=0.2,                          # Shear distortion
    zoom_range=0.25,                          # Zoom in/out
    horizontal_flip=True,                     # Mirror image horizontally
    brightness_range=[0.65, 1.35],            # Simulate lightning changes
    channel_shift_range=25.0,                 # Alter colors slightly
    fill_mode='reflect'                       # Fill empty space with reflection
)
```
* **Image Augmentation**: Because we only have 80 images per class, the model could easily memorize them. Real-time augmentation randomly modifies the images *during* training so the model never sees the exact same image twice.
* **`fill_mode='reflect'`**: Fills empty pixels created by rotations/shifts by mirroring the image borders, which is cleaner than stretching the edges.
* **`validation_split=0.2`**: Split the dataset into 80% training (512 images) and 20% validation (128 images) to measure accuracy on unseen data.

---

### 4. Custom Model Architecture
Our neural network architecture is built inside `build_model()`:
```python
def build_model(num_classes):
    base = EfficientNetB0(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,                  # Exclude original 1000-class classifier
        weights='imagenet'
    )
    base.trainable = False                  # Freeze backbone initially

    x = base.output
    x = GlobalAveragePooling2D()(x)         # Compresses 7x7x1280 grid into a 1280-vector
    x = BatchNormalization()(x)             # Scales values for stable activations
    x = Dense(256, activation='relu')(x)    # Learns complex feature relations
    x = Dropout(0.4)(x)                     # Turns off 40% of neurons randomly
    x = Dense(128, activation='relu')(x)    # Refines features
    x = Dropout(0.3)(x)                     # Turns off 30% of neurons
    output = Dense(num_classes, activation='softmax')(x) # Class probabilities
    
    model = Model(inputs=base.input, outputs=output)
    return model, base
```
* **`GlobalAveragePooling2D`**: Collapses the spatial dimensions of the network into a single vector of features.
* **`Dropout(0.4)` & `Dropout(0.3)`**: A regularization technique. By forcing the network to make predictions without 40% of its connections, it stops relying on specific individual paths and builds a robust, general-purpose understanding.
* **`softmax` Output**: Ensures the sum of output probabilities for all 8 categories adds up to exactly $1.0$ ($100\%$).

---

### 5. Resumable Training State & Callbacks
To prevent losing progress during long training sessions, we implement the following:
* **`SaveStateCallback`**: A custom Keras callback that saves the current phase, epoch index, and best accuracy to a small JSON file (`training_state.json`) after *every single epoch*. If the process crashes, the next run reads this file and resumes exactly where it left off.
* **`ModelCheckpoint`**:
  * Saves the absolute best model weights to `models/best_trained.h5` whenever the validation accuracy increases.
  * Saves temporary checkpoint weights to `checkpoints/checkpoint.weights.h5` every single epoch for recovery.
* **`EarlyStopping(patience=15)`**: If validation accuracy does not improve for 15 consecutive epochs, it halts training and loads the best weights to save time and prevent overfitting.
* **`ReduceLROnPlateau(factor=0.5, patience=7)`**: If the loss plateaus for 7 epochs, it divides the learning rate by $2$ to make smaller, more precise updates to settle the weights.

---

### 6. Phase 1 & Phase 2 Training Runs
* **Phase 1**: Trains the custom layers.
  ```python
  model.compile(optimizer=Adam(learning_rate=LR_HEAD), loss='categorical_crossentropy', metrics=['accuracy'])
  model.fit(train_gen, validation_data=val_gen, callbacks=get_callbacks(phase=1), ...)
  ```
* **Phase 2**: Unfreezes the final 50 layers of the EfficientNet base (while keeping `BatchNormalization` layers frozen to avoid destabilizing statistical variables) and trains with a lower learning rate.
  ```python
  for layer in base_model.layers[FINE_TUNE_AT:]:
      if "BatchNormalization" not in layer.__class__.__name__:
          layer.trainable = True
  ```

---

## 🖥️ Part 2 — Understanding `app.py` (The Interactive Web Interface)

The interactive application is built with **Streamlit** (a framework for converting Python scripts into reactive web pages) and **Plotly** (for rendering beautiful, responsive data visualizations).

### 1. Page Configuration & Premium CSS Injection
```python
st.set_page_config(
    page_title="WasteVision – AI Waste Classifier",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)
```
To create a stunning user experience that matches modern glassmorphic UI design, we inject custom HTML/CSS into the page using `st.markdown(..., unsafe_allow_html=True)`:
* **`linear-gradient` background**: Fades from a deep slate-navy `#0a1628` to a dark midnight `#1a1a2e`.
* **Sidebar styling**: Uses a translucent border and backdrop blur for a frosted-glass effect.
* **Header / Top Bar customization**:
  ```css
  header[data-testid="stHeader"] {
      background: rgba(10, 22, 40, 0.65) !important;
      backdrop-filter: blur(12px) !important;
      border-bottom: 1px solid rgba(255, 255, 255, 0.08) !important;
  }
  ```
  This removes Streamlit's default dark top bar, replacing it with a blurred glassmorphic header matching the app's overall theme.

---

### 2. Metadata Dictionary (`WASTE_META`)
To map raw class names to actionable user insights, we use a dictionary defining emojis, custom colors, and disposal tips for each of the 8 waste classes:
```python
WASTE_META = {
    "Biodegradable": {
        "emoji": "🍂", "color": "#43e97b",
        "tip": "Compost it! Use a compost bin or send to organic waste collection.",
        "examples": "Food scraps, vegetable peels, fruit waste, leaves"
    },
    ...
}
```

---

### 3. Model Loading & Image Preprocessing
We cache the model loading using `@st.cache_resource` so that TensorFlow only reads the large `.keras`/`.h5` file from disk once (when the app boots):
```python
@st.cache_resource(show_spinner="♻️ Loading model …")
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)
```
Before feeding the user's uploaded image or webcam snapshot to the model, we normalize it to match the shape and intensity ranges used during model training:
```python
def preprocess_image(image: Image.Image):
    img = image.resize((IMG_SIZE, IMG_SIZE)) # Resize to 224x224
    img_array = np.array(img)
    # Ensure 3 color channels (RGB)
    if img_array.ndim == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
    # Standardize image array to model input specifications
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)
```

---

### 4. Interactive Tabs
The app is organized into two tabs:
1. **🔍 Classify Waste**:
   * Let's the user choose between **Upload Image** or **Use Webcam**.
   * Resizes, pre-processes, and classifies the image.
   * Renders the top prediction in a styled glass card displaying confidence percentage, disposal tip, and common examples.
   * **Full Probability Breakdown**: Renders a centered, horizontal bar chart powered by Plotly, where each bar's color matches the respective class theme color:
     ```python
     col_space_l, col_chart, col_space_r = st.columns([1, 3, 1])
     with col_chart:
         # Bar chart generated using Plotly Graph Objects (go.Bar)
         st.plotly_chart(fig, use_container_width=True)
     ```
2. **🧠 Model & Network Info**:
   * Reads training metadata from `model_info.json`.
   * Displays key training parameters in a horizontal grid of metrics (Accuracy, Classes, Input Size, Backbone).
   * Explains network parameters, dropout rates, layer architectures, and data augmentation variables.
   * Renders **interactive training curves** (Loss and Accuracy over epochs) using Plotly charts.
   * Displays an interactive waste reference table.

---

## 🛠️ Summary of Training Performance

During training, the system outputs the following summary table to summarize performance stats:

* **Augmentation Strategy**: Online (randomized brightness, rotation, shear, shifts on runtime).
* **Dataset Size**: 640 images (~80 images per class).
* **Backbone**: EfficientNetB0 (highly parameter-efficient, advanced compound-scaled CNN).
* **Target Accuracy**: $90.00\%$
* **Achieved Validation Accuracy**: $\approx 75.78\%$ 
* **Key Saving Paths**:
  * Best Epoch Checkpoint: `models/best_trained.h5`
  * Serialized Training History: `model_info.json`
  * Streamlit Production Model: `models/best_trained.keras`
