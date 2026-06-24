"""
WasteVision – AI Waste Segregation Classifier
Streamlit Application

Usage:
    streamlit run app.py
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import json
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
import plotly.graph_objects as go
import plotly.express as px

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WasteVision – AI Waste Classifier",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Gradient background ── */
.stApp {
    background: linear-gradient(135deg, #0a1628, #132743, #1a1a2e);
    color: #f0f0f0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.04);
    border-right: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(12px);
}

/* ── Hero title ── */
.hero-title {
    font-size: 2.8rem;
    font-weight: 900;
    background: linear-gradient(90deg, #43e97b, #38f9d7, #4facfe);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 0;
    letter-spacing: -1px;
}
.hero-sub {
    text-align: center;
    color: rgba(255,255,255,0.50);
    font-size: 1rem;
    margin-top: 4px;
    margin-bottom: 28px;
}

/* ── Cards ── */
.glass-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 16px;
    padding: 24px;
    backdrop-filter: blur(10px);
    margin-bottom: 20px;
}

/* ── Result boxes ── */
.result-name {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #43e97b, #38f9d7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
}
.result-confidence {
    text-align: center;
    font-size: 1.6rem;
    font-weight: 700;
    color: #43e97b;
}
.result-label {
    text-align: center;
    color: rgba(255,255,255,0.45);
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 6px;
}

/* ── Metric boxes ── */
.metric-box {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 12px;
    padding: 18px 22px;
    text-align: center;
    margin-bottom: 12px;
}
.metric-val {
    font-size: 1.8rem;
    font-weight: 800;
    color: #4facfe;
}
.metric-lbl {
    font-size: 0.75rem;
    color: rgba(255,255,255,0.40);
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.08); }

/* ── Button overrides ── */
.stButton > button {
    border-radius: 10px;
    background: linear-gradient(90deg, #43e97b, #38f9d7);
    border: none;
    color: #0a1628;
    font-weight: 700;
    padding: 0.5rem 1.6rem;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

/* ── Radio ── */
.stRadio > label { color: rgba(255,255,255,0.7); }

/* ── Tab styling ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
    background: transparent;
}
.stTabs [data-baseweb="tab"] {
    background: rgba(255,255,255,0.05);
    border-radius: 10px;
    color: rgba(255,255,255,0.55);
    font-weight: 600;
    padding: 8px 20px;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, #43e97b22, #38f9d722);
    border-bottom: 2px solid #43e97b;
    color: white;
}

/* ── Waste type emoji badges ── */
.waste-badge {
    display: inline-block;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 10px;
    padding: 8px 14px;
    margin: 4px;
    font-size: 0.85rem;
}

/* ── Header / Top Bar ── */
header[data-testid="stHeader"] {
    background: rgba(10, 22, 40, 0.65) !important;
    backdrop-filter: blur(12px) !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08) !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Constants ────────────────────────────────────────────────────────────────
MODEL_PATH = "models/best_trained.keras"
MODEL_H5_PATH = "models/best_trained.h5"
INFO_PATH = "model_info.json"
IMG_SIZE = 224

# Waste type metadata: emoji, colour, disposal tip
WASTE_META = {
    "Biodegradable": {
        "emoji": "🍂", "color": "#43e97b",
        "tip": "Compost it! Use a compost bin or send to organic waste collection.",
        "examples": "Food scraps, vegetable peels, fruit waste, leaves"
    },
    "E-Waste": {
        "emoji": "💻", "color": "#4facfe",
        "tip": "Take to an e-waste recycling centre. Never throw in regular trash!",
        "examples": "Old phones, keyboards, circuit boards, cables"
    },
    "Glass": {
        "emoji": "🍶", "color": "#a18cd1",
        "tip": "Rinse and place in glass recycling bin. Separate by colour if required.",
        "examples": "Bottles, jars, broken glassware"
    },
    "Hazardous": {
        "emoji": "☣️", "color": "#f5576c",
        "tip": "Handle with care! Take to a hazardous waste facility. Never mix with regular waste.",
        "examples": "Batteries, paint cans, medical syringes, chemicals"
    },
    "Metal": {
        "emoji": "🥫", "color": "#fda085",
        "tip": "Crush cans and place in metal recycling bin. Remove labels if possible.",
        "examples": "Soda cans, tin cans, scrap metal, aluminium foil"
    },
    "Paper": {
        "emoji": "📄", "color": "#f9d423",
        "tip": "Keep dry and place in paper recycling bin. Avoid contaminated paper.",
        "examples": "Newspapers, cardboard, office paper, magazines"
    },
    "Plastic": {
        "emoji": "🧴", "color": "#fa709a",
        "tip": "Check the recycling number. Rinse containers before recycling.",
        "examples": "Bottles, bags, cups, food containers"
    },
    "Textile": {
        "emoji": "👕", "color": "#38f9d7",
        "tip": "Donate if wearable! Otherwise, take to a textile recycling centre.",
        "examples": "Old clothes, fabric scraps, rags, curtains"
    },
}


# ─── Helper Functions ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="♻️ Loading model …")
def load_model():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    elif os.path.exists(MODEL_H5_PATH):
        return tf.keras.models.load_model(MODEL_H5_PATH)
    else:
        st.error("❌ Model file not found! Please run train.py first.")
        st.stop()


@st.cache_data(show_spinner=False)
def load_model_info():
    if os.path.exists(INFO_PATH):
        with open(INFO_PATH, 'r') as f:
            return json.load(f)
    return None


def preprocess_image(image: Image.Image):
    """Resize and preprocess an image for EfficientNetB0."""
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img)
    if img_array.ndim == 2:  # grayscale
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[2] == 4:  # RGBA
        img_array = img_array[:, :, :3]
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)


def classify_image(model, image: Image.Image, class_names: list):
    """Run inference and return sorted predictions."""
    processed = preprocess_image(image)
    predictions = model.predict(processed, verbose=0)[0]
    results = []
    for i, name in enumerate(class_names):
        results.append({
            "class": name,
            "confidence": float(predictions[i]),
            "emoji": WASTE_META.get(name, {}).get("emoji", ""),
            "color": WASTE_META.get(name, {}).get("color", "#ffffff"),
        })
    results.sort(key=lambda x: x["confidence"], reverse=True)
    return results


# ─── Load Assets ──────────────────────────────────────────────────────────────
model = load_model()
info = load_model_info()
class_names = info["class_names"] if info else [
    "Biodegradable", "E-Waste", "Glass", "Hazardous",
    "Metal", "Paper", "Plastic", "Textile"
]


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">♻️ WasteVision</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">AI-Powered Waste Segregation Classifier • Upload or Capture • Instant Results</div>', unsafe_allow_html=True)

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab_classify, tab_model = st.tabs(["🔍 Classify Waste", "🧠 Model & Network Info"])

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1: CLASSIFY WASTE
# ══════════════════════════════════════════════════════════════════════════════
with tab_classify:
    st.markdown("---")

    # ── Input method selector ──
    col_input_left, col_input_right = st.columns([1, 1])

    with col_input_left:
        input_method = st.radio(
            "📸 Choose input method",
            ["Upload Image", "Use Webcam"],
            horizontal=True
        )

    uploaded_image = None

    with col_input_right:
        if input_method == "Upload Image":
            uploaded_file = st.file_uploader(
                "Drop an image here", type=["jpg", "jpeg", "png", "webp"],
                label_visibility="collapsed"
            )
            if uploaded_file:
                uploaded_image = Image.open(uploaded_file).convert("RGB")
        else:
            camera_photo = st.camera_input("📷 Capture from webcam")
            if camera_photo:
                uploaded_image = Image.open(camera_photo).convert("RGB")

    # ── Run Classification ──
    if uploaded_image is not None:
        st.markdown("---")

        results = classify_image(model, uploaded_image, class_names)
        top = results[0]
        top_meta = WASTE_META.get(top["class"], {})

        col_img, col_result = st.columns([1, 1.3], gap="large")

        # ── Left: Show uploaded image ──
        with col_img:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.image(uploaded_image, use_container_width=True, caption="Uploaded Image")
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Right: Show prediction result ──
        with col_result:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)

            st.markdown('<div class="result-label">Identified As</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="result-name">{top_meta.get("emoji", "")} {top["class"]}</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                f'<div class="result-confidence">{top["confidence"]*100:.1f}% Confidence</div>',
                unsafe_allow_html=True
            )

            st.markdown("---")

            # Disposal tip
            st.markdown(f"""
            <div style="background: rgba(67,233,123,0.08); border: 1px solid rgba(67,233,123,0.2);
                        border-radius: 10px; padding: 14px 18px; margin-bottom: 12px;">
                <span style="font-weight: 700; color: #43e97b;">💡 Disposal Tip</span><br>
                <span style="color: rgba(255,255,255,0.75); font-size: 0.9rem;">
                    {top_meta.get("tip", "Please dispose responsibly.")}
                </span>
            </div>
            """, unsafe_allow_html=True)

            # Examples
            st.markdown(f"""
            <div style="color: rgba(255,255,255,0.40); font-size: 0.8rem; margin-top: 4px;">
                <b>Common examples:</b> {top_meta.get("examples", "")}
            </div>
            """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

        # ── Full Probability Breakdown ──
        st.markdown("<h3 style='text-align: center; margin-top: 30px; margin-bottom: 20px;'>📊 Full Probability Breakdown</h3>", unsafe_allow_html=True)

        col_space_l, col_chart, col_space_r = st.columns([1, 3, 1])

        with col_chart:
            # Horizontal bar chart with plotly
            labels = [f"{r['emoji']} {r['class']}" for r in results]
            values = [r["confidence"] * 100 for r in results]
            colors = [r["color"] for r in results]

            fig = go.Figure(go.Bar(
                x=values,
                y=labels,
                orientation='h',
                marker=dict(
                    color=colors,
                    line=dict(width=0),
                    opacity=0.85,
                ),
                text=[f"{v:.1f}%" for v in values],
                textposition='outside',
                textfont=dict(color='white', size=12),
            ))
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='rgba(255,255,255,0.7)', family='Inter'),
                xaxis=dict(
                    title="Confidence (%)", range=[0, max(values) * 1.25],
                    gridcolor='rgba(255,255,255,0.06)',
                    tickfont=dict(color='rgba(255,255,255,0.5)')
                ),
                yaxis=dict(
                    autorange="reversed",
                    tickfont=dict(color='rgba(255,255,255,0.8)', size=13)
                ),
                height=380,
                margin=dict(l=10, r=40, t=20, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

    else:
        # ── Empty state ──
        st.markdown("""
        <div class="glass-card" style="text-align: center; padding: 60px 20px;">
            <div style="font-size: 4rem; margin-bottom: 12px;">♻️</div>
            <div style="font-size: 1.3rem; font-weight: 700; color: rgba(255,255,255,0.7); margin-bottom: 8px;">
                Upload or capture an image to classify waste
            </div>
            <div style="color: rgba(255,255,255,0.35); font-size: 0.9rem;">
                Supports JPG, PNG, WEBP • Works with photos of waste items
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Show supported classes
        st.markdown("#### 🏷️ Supported Waste Classes")
        badge_html = ""
        for name, meta in WASTE_META.items():
            badge_html += f'<span class="waste-badge">{meta["emoji"]} {name}</span>'
        st.markdown(f'<div style="text-align: center; margin-top: 8px;">{badge_html}</div>',
                    unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2: MODEL & NEURAL NETWORK INFO
# ══════════════════════════════════════════════════════════════════════════════
with tab_model:
    st.markdown("---")

    if info is None:
        st.warning("⚠️ model_info.json not found. Please run train.py first.")
    else:
        # ── Overview Metrics ──
        st.markdown("### 📋 Model Overview")
        m1, m2, m3, m4 = st.columns(4)

        with m1:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-val">{info.get("model_architecture", "N/A").split("+")[0].strip()}</div>
                <div class="metric-lbl">Backbone</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-val">{info.get("num_classes", 8)}</div>
                <div class="metric-lbl">Classes</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-val">{info.get("best_val_accuracy", 0)*100:.1f}%</div>
                <div class="metric-lbl">Best Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-val">{info.get("img_size", 224)}×{info.get("img_size", 224)}</div>
                <div class="metric-lbl">Input Size</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # ── Architecture Details ──
        col_arch, col_hyper = st.columns(2, gap="large")

        with col_arch:
            st.markdown("### 🏗️ Network Architecture")
            st.markdown(f"""
            <div class="glass-card">
                <div style="font-size: 0.9rem; line-height: 1.8;">
                    <b style="color: #4facfe;">Model:</b> {info.get("model_architecture", "N/A")}<br>
                    <b style="color: #4facfe;">Input:</b> {info.get("img_size", 224)}×{info.get("img_size", 224)}×3 RGB Image<br>
                    <b style="color: #4facfe;">Backbone:</b> EfficientNetB0 (ImageNet pre-trained)<br>
                    <b style="color: #4facfe;">Head:</b> GAP → BatchNorm → Dense(256) → Dropout(0.4) → Dense(128) → Dropout(0.3) → Dense(8, softmax)<br>
                    <b style="color: #4facfe;">Total Params:</b> ~4.4M (16.85 MB)<br>
                    <b style="color: #4facfe;">Trainable Params:</b> ~2.87M (10.96 MB)<br>
                    <b style="color: #4facfe;">Output:</b> 8-class softmax probability distribution
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### 🔬 What is EfficientNetB0?")
            st.markdown("""
            <div class="glass-card" style="font-size: 0.85rem; color: rgba(255,255,255,0.7); line-height: 1.7;">
                <b>EfficientNet</b> is a family of convolutional neural networks developed by Google Brain.
                It uses <b>compound scaling</b> — simultaneously scaling depth, width, and resolution — to achieve
                state-of-the-art accuracy with far fewer parameters than older architectures like ResNet or VGG.<br><br>
                <b>Key innovations:</b><br>
                • <b>MBConv blocks</b> — mobile inverted bottleneck convolutions for efficiency<br>
                • <b>Squeeze-and-Excitation (SE)</b> — channel attention that helps the network focus on important features<br>
                • <b>Swish activation</b> — smooth, non-monotonic activation function that outperforms ReLU<br>
                • <b>Compound scaling</b> — balanced scaling of network dimensions<br><br>
                EfficientNetB0 was pre-trained on <b>ImageNet</b> (1.4M images, 1000 classes), giving it rich
                feature representations that transfer well to waste classification.
            </div>
            """, unsafe_allow_html=True)

        with col_hyper:
            st.markdown("### ⚙️ Training Configuration")
            hp = info.get("hyperparameters", {})
            st.markdown(f"""
            <div class="glass-card">
                <div style="font-size: 0.9rem; line-height: 1.8;">
                    <b style="color: #43e97b;">Training Strategy:</b> Two-Phase Transfer Learning<br>
                    <b style="color: #43e97b;">Phase 1:</b> Train head only ({hp.get("epochs_head", "N/A")} epochs, LR={hp.get("lr_head", "N/A")})<br>
                    <b style="color: #43e97b;">Phase 2:</b> Fine-tune last {hp.get("fine_tune_layers", "N/A")} backbone layers ({hp.get("epochs_fine", "N/A")} epochs, LR={hp.get("lr_fine", "N/A")})<br>
                    <b style="color: #43e97b;">Batch Size:</b> {hp.get("batch_size", "N/A")}<br>
                    <b style="color: #43e97b;">Label Smoothing:</b> {hp.get("label_smoothing", "N/A")}<br>
                    <b style="color: #43e97b;">Augmentation:</b> {info.get("augmentation_strategy", "N/A")}
                </div>
            </div>
            """, unsafe_allow_html=True)

            aug = hp.get("augmentation", {})
            if aug:
                st.markdown("#### 🎨 Data Augmentation")
                st.markdown(f"""
                <div class="glass-card">
                    <div style="font-size: 0.85rem; line-height: 1.7; color: rgba(255,255,255,0.7);">
                        <b>Rotation:</b> ±{aug.get("rotation_range", "N/A")}°<br>
                        <b>Width Shift:</b> ±{aug.get("width_shift_range", "N/A")}<br>
                        <b>Height Shift:</b> ±{aug.get("height_shift_range", "N/A")}<br>
                        <b>Shear:</b> {aug.get("shear_range", "N/A")}<br>
                        <b>Zoom:</b> ±{aug.get("zoom_range", "N/A")}<br>
                        <b>Horizontal Flip:</b> {"✅" if aug.get("horizontal_flip") else "❌"}<br>
                        <b>Brightness:</b> {aug.get("brightness_range", "N/A")}<br>
                        <b>Channel Shift:</b> ±{aug.get("channel_shift_range", "N/A")}<br>
                        <b>Fill Mode:</b> {aug.get("fill_mode", "N/A")}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # ── Training History Charts ──
        history = info.get("training_history", {})
        if history.get("accuracy"):
            st.markdown("### 📈 Training History")

            col_acc, col_loss = st.columns(2, gap="large")

            epochs = list(range(1, len(history["accuracy"]) + 1))

            with col_acc:
                fig_acc = go.Figure()
                fig_acc.add_trace(go.Scatter(
                    x=epochs, y=[a * 100 for a in history["accuracy"]],
                    mode='lines', name='Train Accuracy',
                    line=dict(color='#4facfe', width=2.5),
                    fill='tozeroy', fillcolor='rgba(79,172,254,0.08)',
                ))
                if history.get("val_accuracy"):
                    fig_acc.add_trace(go.Scatter(
                        x=epochs[:len(history["val_accuracy"])],
                        y=[a * 100 for a in history["val_accuracy"]],
                        mode='lines', name='Val Accuracy',
                        line=dict(color='#43e97b', width=2.5),
                        fill='tozeroy', fillcolor='rgba(67,233,123,0.08)',
                    ))
                fig_acc.update_layout(
                    title=dict(text="Accuracy", font=dict(color='white', size=16)),
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='rgba(255,255,255,0.6)', family='Inter'),
                    xaxis=dict(title="Epoch", gridcolor='rgba(255,255,255,0.05)'),
                    yaxis=dict(title="Accuracy (%)", gridcolor='rgba(255,255,255,0.05)'),
                    legend=dict(bgcolor='rgba(0,0,0,0)'),
                    height=350, margin=dict(l=10, r=10, t=40, b=40),
                )
                st.plotly_chart(fig_acc, use_container_width=True)

            with col_loss:
                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(
                    x=epochs, y=history["loss"],
                    mode='lines', name='Train Loss',
                    line=dict(color='#fa709a', width=2.5),
                    fill='tozeroy', fillcolor='rgba(250,112,154,0.08)',
                ))
                if history.get("val_loss"):
                    fig_loss.add_trace(go.Scatter(
                        x=epochs[:len(history["val_loss"])],
                        y=history["val_loss"],
                        mode='lines', name='Val Loss',
                        line=dict(color='#fda085', width=2.5),
                        fill='tozeroy', fillcolor='rgba(253,160,133,0.08)',
                    ))
                fig_loss.update_layout(
                    title=dict(text="Loss", font=dict(color='white', size=16)),
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='rgba(255,255,255,0.6)', family='Inter'),
                    xaxis=dict(title="Epoch", gridcolor='rgba(255,255,255,0.05)'),
                    yaxis=dict(title="Loss", gridcolor='rgba(255,255,255,0.05)'),
                    legend=dict(bgcolor='rgba(0,0,0,0)'),
                    height=350, margin=dict(l=10, r=10, t=40, b=40),
                )
                st.plotly_chart(fig_loss, use_container_width=True)

        # ── Class Info Grid ──
        st.markdown("---")
        st.markdown("### 🏷️ Waste Classes Reference")

        cols = st.columns(4, gap="medium")
        for i, (name, meta) in enumerate(WASTE_META.items()):
            with cols[i % 4]:
                st.markdown(f"""
                <div class="glass-card" style="text-align: center; min-height: 160px;">
                    <div style="font-size: 2.2rem; margin-bottom: 6px;">{meta["emoji"]}</div>
                    <div style="font-weight: 700; font-size: 1rem; color: {meta["color"]}; margin-bottom: 6px;">
                        {name}
                    </div>
                    <div style="font-size: 0.78rem; color: rgba(255,255,255,0.45); line-height: 1.5;">
                        {meta["examples"]}
                    </div>
                </div>
                """, unsafe_allow_html=True)


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ♻️ WasteVision")
    st.markdown("""
    <div style="color: rgba(255,255,255,0.5); font-size: 0.85rem; line-height: 1.6;">
        AI-powered waste segregation classifier built with
        <b>EfficientNetB0</b> transfer learning and <b>Streamlit</b>.<br><br>
        Upload a photo of any waste item and the model will classify it
        into one of 8 categories with a confidence score.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    if info:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-val">{info.get("best_val_accuracy", 0)*100:.1f}%</div>
            <div class="metric-lbl">Model Accuracy</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-val">640</div>
            <div class="metric-lbl">Training Images</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-val">4.4M</div>
            <div class="metric-lbl">Parameters</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="color: rgba(255,255,255,0.3); font-size: 0.75rem; text-align: center;">
        Built with ❤️ using TensorFlow & Streamlit
    </div>
    """, unsafe_allow_html=True)
