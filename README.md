# ♻️ WasteVision — AI-Powered Waste Segregation Classifier

WasteVision is a high-performance deep learning application designed to classify waste items into 8 categories and provide actionable disposal tips. It uses **Transfer Learning with EfficientNetB0** for feature extraction and a premium **Streamlit** dark-themed web interface for real-time predictions.

For a detailed, line-by-line explanation of the entire codebase from absolute scratch, check out [PROJECT_EXPLAINED.md](file:///d:/code/ML%20&%20NN/NN%20Project/WasteSegrigation-strmlit/PROJECT_EXPLAINED.md).

---

## 🚀 Features

* **Instant Classification**: Upload any photo (`JPG`, `PNG`, `WEBP`) or capture one live using your **Webcam**.
* **Smart Recommendations**: Every classification includes an eco-friendly **Disposal Tip** and examples of matching waste types.
* **Centered Confidence Breakdown**: Interactive horizontal bar charts showing prediction probabilities for all categories.
* **Premium Glassmorphic UI**: High-fidelity dark mode interface with smooth animations, custom fonts, and themed accents.
* **Analytics Dashboard**: In-app training statistics, architecture specifications, and historical loss/accuracy validation curves.
* **Resumable Training**: Training pipeline checks checkpoints and training state JSON files, allowing you to pause and resume training epochs without losing progress.

---

## 📂 Project Directory Structure

```text
WasteSegrigation-strmlit/
├── Dataset/                   # Organized folders per waste category (80 images each)
│   ├── Biodegradable/
│   ├── E-Waste/
│   ├── Glass/
│   ├── Hazardous/
│   ├── Metal/
│   ├── Paper/
│   ├── Plastic/
│   └── Textile/
├── models/                    # Saved neural network models
│   ├── best_trained.h5        # Best validation model (.h5 format)
│   ├── best_trained.keras     # Production model (.keras format)
│   └── final_result.h5        # Final epoch model weights
├── checkpoints/               # Temporary files for resumable training state
│   └── checkpoint.weights.h5  # Latest epoch weights
├── app.py                     # Streamlit web application
├── train.py                   # Model training and fine-tuning pipeline
├── model_info.json            # Serialized model hyperparameters & training history
├── training_state.json        # Auto-saved epoch state for resumable training
├── count_images.py            # Diagnostic script to verify dataset file counts
├── download_images.py         # Automated dataset scraper script
├── PROJECT_EXPLAINED.md       # Full mathematical & code documentation
└── README.md                  # This file
```

---

## 🏷️ Supported Classes

| Icon | Category | Examples | Disposal Tip |
| :---: | :--- | :--- | :--- |
| 🍂 | **Biodegradable** | Food scraps, leaves, fruit peels | Compost it in a bin or organic bin. |
| 💻 | **E-Waste** | Old phones, keyboards, cables | Drop off at certified e-waste facilities. |
| 🍶 | **Glass** | Bottles, jars, broken glassware | Rinse and place in glass recycling bins. |
| ☣️ | **Hazardous** | Batteries, paint cans, chemicals | Bring to hazardous waste drop-off centers. |
| 🥫 | **Metal** | Soda cans, foil, scrap metal | Crush and place in metal recycling bins. |
| 📄 | **Paper** | Newspapers, cardboard, magazines | Keep dry and place in paper recycling. |
| 🧴 | **Plastic** | Bottles, containers, plastic bags | Rinse and check the plastic code before recycling. |
| 👕 | **Textile** | Old clothes, fabric rags, curtains | Donate if usable, or take to textile recycling. |

---

## 💻 Installation & Getting Started

### 1. Requirements & Dependencies
Ensure you have Python 3.8+ installed. Install the required python packages:
```bash
pip install tensorflow streamlit numpy pillow plotly
```

### 2. Launch the Application
Run the Streamlit app locally:
```bash
streamlit run app.py
```
This opens the web interface in your default browser at **http://localhost:8501**.

### 3. Run Training (Optional)
If you wish to re-train the model or continue training from current checkpoints:
```bash
python train.py
```
The training script will automatically flow images from the `Dataset/` directory, perform online augmentations, train in two phases, and save model files.

---

## 🧠 Model Architecture & Training Details

WasteVision utilizes a **Two-Phase Transfer Learning** strategy built around **EfficientNetB0**:

1. **Backbone Feature Extractor**: EfficientNetB0 pre-trained on ImageNet (1.4 million images).
2. **Custom Classification Head**:
   ```text
   GlobalAveragePooling2D ──► BatchNormalization ──► Dense(256, ReLU) ──► Dropout(0.4) ──► Dense(128, ReLU) ──► Dropout(0.3) ──► Dense(8, Softmax)
   ```
3. **Training Strategy**:
   * **Phase 1 (Epochs 0-40)**: Backbone frozen, custom head trained with Adam ($LR = 3 \times 10^{-4}$).
   * **Phase 2 (Epochs 40-90)**: Last 50 backbone layers unfrozen (with BatchNormalization frozen), trained with a low learning rate ($LR = 2 \times 10^{-5}$).
   * **Augmentation**: Applied on-the-fly (rotation, shift, zoom, brightness variance, color jitter).
   * **Results**: Reached validation accuracy of **75.78%** on 640 images.
