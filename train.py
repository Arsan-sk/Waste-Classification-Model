"""
train.py -- 8-Class Waste Image Classifier
==========================================
Transfer Learning with EfficientNetB0 (ImageNet weights).
Two-phase training:
  Phase 1: Train custom head only (backbone frozen).
  Phase 2: Fine-tune last 50 backbone layers (with BatchNormalization layers frozen).

Online (runtime) augmentation -- no extra images saved to disk.
Fully resumable from any interruption via training_state.json + checkpoints.

Usage:
    python train.py
"""

# ==============================================================================
#  IMPORTS
# ==============================================================================
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import json
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import (
    Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


# ==============================================================================
#  STEP 1 -- CONFIGURATION
# ==============================================================================
IMG_SIZE         = 224          # 224x224 for better feature detail on subtle waste types
BATCH_SIZE       = 16           # small batch suits small dataset
EPOCHS_HEAD      = 40           # Phase 1: train custom head thoroughly
EPOCHS_FINE      = 50           # Phase 2: fine-tune backbone layers
LR_HEAD          = 3e-4         # Phase 1 learning rate
LR_FINE          = 2e-5         # Phase 2 LR -- very small to avoid catastrophic forgetting
FINE_TUNE_AT     = -50          # Unfreeze last 50 layers of EfficientNetB0 in Phase 2
SEED             = 42
NUM_CLASSES      = 8
TARGET_ACCURACY  = 0.90         # 90% is goal; 80% is minimum acceptable
LABEL_SMOOTHING  = 0.1          # soften one-hot labels to fight overconfidence
DATASET_DIR      = "Dataset"
MODELS_DIR       = "models"
CKPT_DIR         = "checkpoints"
CHECKPOINT_PATH  = "checkpoints/checkpoint.weights.h5"
STATE_FILE       = "training_state.json"
MODEL_INFO_FILE  = "model_info.json"
BEST_MODEL_PATH  = "models/best_trained.h5"
FINAL_MODEL_PATH = "models/final_result.h5"


# ==============================================================================
#  STEP 2 -- DIRECTORY SETUP & SEEDING
# ==============================================================================
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CKPT_DIR,   exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("=" * 60)
print("  WASTE IMAGE CLASSIFIER -- EfficientNetB0 Transfer Learning")
print("=" * 60)
print(f"[SETUP] Output directories created: {MODELS_DIR}/, {CKPT_DIR}/")
print(f"[SETUP] Random seed set to {SEED} for reproducibility")


# ==============================================================================
#  STEP 3 -- LOAD TRAINING STATE (resume capability)
# ==============================================================================
def load_training_state():
    """
    Returns dict:
    {
      "phase": 1 or 2,           # which training phase to resume
      "initial_epoch": int,      # epoch number to resume from (0 = fresh start)
      "best_val_accuracy": float # best accuracy seen so far across all runs
    }
    """
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            state = json.load(f)
        print(f"[RESUME] Resuming from Phase {state['phase']}, "
              f"Epoch {state['initial_epoch']}, "
              f"Best Acc so far: {state['best_val_accuracy']:.4f}")
        return state
    else:
        print("[STATE] No saved state found. Starting fresh from Phase 1, Epoch 0.")
        return {"phase": 1, "initial_epoch": 0, "best_val_accuracy": 0.0}


state = load_training_state()


# ==============================================================================
#  STEP 4 -- DATA GENERATORS (with moderate online augmentation)
# ==============================================================================
print("\n[DATA] Setting up data generators...")

# CRITICAL: MobileNetV2 was trained with inputs in [-1, 1] range.
# Previous runs used rescale=1/255 (-> [0,1]) which is WRONG for transfer learning.
# preprocess_input maps [0,255] -> [-1,1], matching ImageNet training distribution.
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # [-1, 1] normalization (NOT rescale!)
    validation_split=0.2,

    # -- Geometric transforms (moderate) ----------------------
    rotation_range=25,           # rotate up to +/-25 deg
    width_shift_range=0.2,       # shift horizontally up to 20%
    height_shift_range=0.2,      # shift vertically up to 20%
    shear_range=0.2,             # moderate shear transformation
    zoom_range=0.25,             # zoom in/out up to 25%
    horizontal_flip=True,        # mirror left-right (valid for symmetric objects)

    # -- Colour / lighting transforms (moderate) --------------
    brightness_range=[0.65, 1.35], # simulate lighting variation
    channel_shift_range=25.0,    # moderate colour jitter

    fill_mode='reflect'          # fill empty pixels with reflection -- better than
                                 # 'nearest' for object images (avoids black borders)
)

# Validation generator -- preprocess ONLY, no augmentation (must represent real-world input)
val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # same [-1, 1] normalization
    validation_split=0.2
)

# Flow generators from Dataset/ directly
train_gen = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    seed=SEED,
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    seed=SEED,
    shuffle=False                # never shuffle validation
)

# Store class names for model_info.json
class_names = list(train_gen.class_indices.keys())

print(f"[DATA] Classes found     : {len(class_names)}")
print(f"[DATA] Class mapping     : {train_gen.class_indices}")
print(f"[DATA] Training samples  : {train_gen.samples}")
print(f"[DATA] Validation samples: {val_gen.samples}")
print(f"[DATA] Augmentation      : ONLINE (live, different every epoch)")


# ==============================================================================
#  STEP 5 -- BUILD MODEL
# ==============================================================================
def build_model(num_classes):
    """
    Build EfficientNetB0 with a deeper classification head.
    Returns (model, base_model) so base_model layers can be unfrozen in Phase 2.

    Head: GAP -> BN -> Dense(256) -> Dropout(0.4) -> Dense(128) -> Dropout(0.3) -> Dense(8, softmax)
    EfficientNetB0's 1280-dim GAP output is very rich. A two-layer head
    with graduated dropout gives enough capacity for 8 waste classes
    while controlling overfitting on a small dataset.
    """
    base = EfficientNetB0(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'       # borrow knowledge from 1.4M ImageNet images
    )
    base.trainable = False       # freeze entire backbone in Phase 1

    x = base.output
    x = GlobalAveragePooling2D()(x)    # flatten 7x7x1280 -> 1280-vector
    x = BatchNormalization()(x)        # stabilise activations
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)               # graduated dropout -- stronger here
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)               # lighter dropout near output
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base.input, outputs=output)
    return model, base


# ==============================================================================
#  STEP 6 -- SaveStateCallback
# ==============================================================================
class SaveStateCallback(tf.keras.callbacks.Callback):
    """
    Saves training state to disk after every single epoch, so any interruption
    (crash, Ctrl+C, power cut) can be resumed from the last completed epoch.
    """
    def __init__(self, phase):
        super().__init__()
        self.phase = phase

    def on_epoch_end(self, epoch, logs=None):
        # For Phase 2, convert absolute epoch back to phase-relative epoch
        if self.phase == 2:
            phase_epoch = epoch - EPOCHS_HEAD + 1  # next epoch within Phase 2
        else:
            phase_epoch = epoch + 1                 # next epoch within Phase 1

        state = {
            "phase": self.phase,
            "initial_epoch": phase_epoch,
            "best_val_accuracy": float(logs.get('val_accuracy', 0.0))
        }
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)
        # Silent save -- no print needed (Keras already prints epoch results)


# ==============================================================================
#  STEP 7 -- CALLBACKS
# ==============================================================================
def get_callbacks(phase):
    """
    Build callback list fresh for each phase with the correct SaveStateCallback.
    """
    return [
        # 1. Save best full model whenever val_accuracy improves
        ModelCheckpoint(
            filepath=BEST_MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,   # save full model (architecture + weights)
            verbose=1
        ),

        # 2. Save weights every epoch for resume (not just best)
        ModelCheckpoint(
            filepath=CHECKPOINT_PATH,
            monitor='val_accuracy',
            save_best_only=False,      # overwrite every epoch -- we want the latest
            save_weights_only=True,    # weights only -- smaller file, faster save
            verbose=0
        ),

        # 3. Early stopping -- halt if no improvement
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,               # halt if no improvement for 10 epochs
            restore_best_weights=True,
            verbose=1
        ),

        # 4. Reduce learning rate on plateau
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,                # multiply LR by 0.5 when stuck (gentler decay)
            patience=7,
            min_lr=1e-7,
            verbose=1
        ),

        # 5. Persist training state after every epoch
        SaveStateCallback(phase=phase)
    ]


# ==============================================================================
#  STEP 8 -- PHASE 1 TRAINING (Custom Head Only)
# ==============================================================================
history_head = None  # will hold Phase 1 history if Phase 1 runs in this session
history_fine = None  # will hold Phase 2 history

if state["phase"] == 1:
    print("\n" + "=" * 60)
    print("  PHASE 1 -- Training Custom Head Only (Backbone Frozen)")
    print("=" * 60)

    model, base_model = build_model(NUM_CLASSES)

    total_params = model.count_params()
    trainable_params = sum(
        tf.keras.backend.count_params(w) for w in model.trainable_weights
    )
    print(f"[MODEL] EfficientNetB0 backbone loaded (ImageNet weights)")
    print(f"[MODEL] Custom head: GAP -> BN -> Dense(256) -> Dense(128) -> Dense({NUM_CLASSES}, softmax)")
    print(f"[MODEL] Total params: {total_params:,}  |  Trainable (Phase 1): {trainable_params:,}")

    if state["initial_epoch"] > 0:
        model.load_weights(CHECKPOINT_PATH)
        print(f"[RESUME] Loaded Phase 1 checkpoint from epoch {state['initial_epoch']}")
    else:
        print("[PHASE 1] Starting fresh -- training custom head only")

    model.compile(
        optimizer=Adam(learning_rate=LR_HEAD),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=['accuracy']
    )

    print(f"[PHASE 1] Training for epochs {state['initial_epoch']+1}–{EPOCHS_HEAD} "
          f"(LR={LR_HEAD})\n")

    history_head = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_HEAD,
        initial_epoch=state["initial_epoch"],
        callbacks=get_callbacks(phase=1)
    )

    # Phase 1 complete -- update state for Phase 2
    phase1_best_acc = max(history_head.history['val_accuracy'])
    state = {
        "phase": 2,
        "initial_epoch": 0,
        "best_val_accuracy": phase1_best_acc
    }
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)

    print(f"\n[PHASE 1] Complete. Best val_accuracy: {phase1_best_acc:.4f}. Moving to Phase 2.")


# ==============================================================================
#  STEP 9 -- PHASE 2 FINE-TUNING (Unfreeze Last N Backbone Layers)
# ==============================================================================
if state["phase"] == 2:
    print("\n" + "=" * 60)
    print(f"  PHASE 2 -- Fine-Tuning Last {abs(FINE_TUNE_AT)} Backbone Layers")
    print("=" * 60)

    # If resuming from a previous run (Phase 1 didn't run this session), rebuild
    if 'model' not in dir() or model is None:
        model, base_model = build_model(NUM_CLASSES)
        print(f"[MODEL] EfficientNetB0 backbone loaded (ImageNet weights)")
        print(f"[MODEL] Custom head: GAP -> BN -> Dense(256) -> Dense(128) -> Dense({NUM_CLASSES}, softmax)")

    # Unfreeze last N layers of the backbone, keeping BatchNormalization layers frozen
    for layer in base_model.layers[FINE_TUNE_AT:]:
        if "BatchNormalization" not in layer.__class__.__name__:
            layer.trainable = True
        else:
            layer.trainable = False
    # All earlier layers remain frozen -- they contain universal low-level features

    if state["initial_epoch"] > 0:
        model.load_weights(CHECKPOINT_PATH)
        print(f"[RESUME] Loaded Phase 2 checkpoint from epoch {state['initial_epoch']}")
    else:
        if os.path.exists(BEST_MODEL_PATH):
            model.load_weights(BEST_MODEL_PATH)
            print(f"[PHASE 2] Loaded best Phase 1 weights from {BEST_MODEL_PATH}")
        else:
            print(f"[PHASE 2] Fine-tuning last {abs(FINE_TUNE_AT)} backbone layers (no Phase 1 weights found)")

    # IMPORTANT -- always recompile after changing trainable layers
    model.compile(
        optimizer=Adam(learning_rate=LR_FINE),   # 50x smaller than Phase 1
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=['accuracy']
    )

    trainable_params = sum(
        tf.keras.backend.count_params(w) for w in model.trainable_weights
    )
    print(f"[PHASE 2] Trainable params now: {trainable_params:,} (backbone partially unfrozen, BN layers frozen)")

    phase2_start_epoch = EPOCHS_HEAD + state["initial_epoch"]
    phase2_end_epoch   = EPOCHS_HEAD + EPOCHS_FINE

    print(f"[PHASE 2] Training for epochs {phase2_start_epoch+1}–{phase2_end_epoch} "
          f"(LR={LR_FINE})\n")

    history_fine = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=phase2_end_epoch,                    # absolute epoch number
        initial_epoch=phase2_start_epoch,           # continue from where Phase 1 left off
        callbacks=get_callbacks(phase=2)
    )


# ==============================================================================
#  STEP 10 -- POST-TRAINING LOGIC
# ==============================================================================
print("\n" + "=" * 60)
print("  POST-TRAINING")
print("=" * 60)

final_val_acc = max(history_fine.history['val_accuracy']) if history_fine else 0.0
phase1_best = (max(history_head.history['val_accuracy'])
               if history_head and history_head.history.get('val_accuracy') else None)

# Always save final_result.h5 -- regardless of accuracy reached
model.save(FINAL_MODEL_PATH)
print(f"[SAVE] final_result.h5 saved -> {FINAL_MODEL_PATH}")
print(f"[SAVE] best_trained.h5 already saved by ModelCheckpoint -> {BEST_MODEL_PATH}")

# Save production models in native Keras format as well
if os.path.exists(BEST_MODEL_PATH):
    try:
        best_model_loaded = tf.keras.models.load_model(BEST_MODEL_PATH)
        best_model_loaded.save("models/best_trained.keras")
        print("[SAVE] best_trained.keras updated from best_trained.h5")
    except Exception as e:
        print(f"[WARN] Failed to save best_trained.keras: {e}")

try:
    model.save("models/final_result.keras")
    print("[SAVE] final_result.keras updated")
except Exception as e:
    print(f"[WARN] Failed to save final_result.keras: {e}")

# Report result
if final_val_acc >= TARGET_ACCURACY:
    print(f"\n[SUCCESS] (target) Target {TARGET_ACCURACY*100:.0f}% REACHED! "
          f"Final: {final_val_acc*100:.2f}%")
    # Clear state -- next run will start fresh
    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)
    print("[STATE] training_state.json cleared. Next run starts fresh.")
else:
    print(f"\n[INFO] Accuracy {final_val_acc*100:.2f}% is below target "
          f"{TARGET_ACCURACY*100:.0f}%.")
    print(f"[INFO] Model saved. Run train.py again to continue improving from checkpoint.")
    # Do NOT delete STATE_FILE -- keep it so next run resumes


# ==============================================================================
#  STEP 11 -- SAVE model_info.json
# ==============================================================================
# Merge histories from both phases into one continuous record
# Guard for the case where Phase 1 ran in a previous session
merged_history = {}
for key in ['accuracy', 'val_accuracy', 'loss', 'val_loss']:
    phase1_values = (history_head.history.get(key, [])
                     if history_head is not None else [])
    phase2_values = (history_fine.history.get(key, [])
                     if history_fine is not None else [])
    merged_history[key] = phase1_values + phase2_values

model_info = {
    "class_names": class_names,
    "num_classes": NUM_CLASSES,
    "img_size": IMG_SIZE,
    "model_architecture": "EfficientNetB0 + Custom Head (256-128)",
    "augmentation_strategy": "Online (runtime) -- different random variation every epoch",
    "hyperparameters": {
        "batch_size": BATCH_SIZE,
        "epochs_head": EPOCHS_HEAD,
        "epochs_fine": EPOCHS_FINE,
        "lr_head": LR_HEAD,
        "lr_fine": LR_FINE,
        "fine_tune_layers": abs(FINE_TUNE_AT),
        "augmentation": {
            "rotation_range": 20,
            "width_shift_range": 0.15,
            "height_shift_range": 0.15,
            "shear_range": 0.15,
            "zoom_range": 0.2,
            "horizontal_flip": True,
            "vertical_flip": False,
            "brightness_range": [0.7, 1.3],
            "channel_shift_range": 20.0,
            "fill_mode": "reflect"
        },
        "label_smoothing": LABEL_SMOOTHING
    },
    "training_history": merged_history,
    "best_val_accuracy": max(merged_history.get('val_accuracy', [0])),
    "final_val_accuracy": final_val_acc,
    "target_accuracy": TARGET_ACCURACY
}

with open(MODEL_INFO_FILE, 'w') as f:
    json.dump(model_info, f, indent=2)
print(f"[SAVE] model_info.json saved -> {MODEL_INFO_FILE}")


# ==============================================================================
#  STEP 12 -- FINAL SUMMARY TABLE
# ==============================================================================
best_overall = max(merged_history.get('val_accuracy', [0]))
target_reached = "YES" if best_overall >= TARGET_ACCURACY else "NO"
phase1_display = (f"{phase1_best*100:.2f}%" if phase1_best is not None
                  else "N/A (ran in previous session)")
phase2_display = f"{final_val_acc*100:.2f}%" if final_val_acc > 0 else "N/A"
total_images = train_gen.samples + val_gen.samples

print("\n")
print("+==========================================================+")
print("|         WASTE CLASSIFIER -- TRAINING SUMMARY             |")
print("+==========================================================+")
print(f"|  Augmentation Strategy   : Online (runtime)             |")
print(f"|  Original Images         : ~{total_images} ({total_images // NUM_CLASSES} per class){' ' * max(0, 14 - len(str(total_images)) - len(str(total_images // NUM_CLASSES)))}|")
print(f"|  Classes                 : {NUM_CLASSES}                             |")
print(f"|  Phase 1 Best Val Acc    : {phase1_display}{' ' * max(0, 28 - len(phase1_display))}|")
print(f"|  Phase 2 Best Val Acc    : {phase2_display}{' ' * max(0, 28 - len(phase2_display))}|")
print(f"|  Target Accuracy         : {TARGET_ACCURACY*100:.2f}%{' ' * 22}|")
print(f"|  Target Reached?         : {target_reached}{' ' * max(0, 28 - len(target_reached))}|")
print(f"|  Best Model Saved        : {BEST_MODEL_PATH}{' ' * max(0, 28 - len(BEST_MODEL_PATH))}|")
print(f"|  Final Model Saved       : {FINAL_MODEL_PATH}{' ' * max(0, 28 - len(FINAL_MODEL_PATH))}|")
print("+==========================================================+")
print("\n[DONE] Training pipeline finished.")
