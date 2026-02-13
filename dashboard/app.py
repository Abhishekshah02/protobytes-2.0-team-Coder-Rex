"""
Streamlit Dashboard with Live Phone Integration
"""

import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import time
import json
import os
import sys
from scipy import signal
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from model import IMUEncoder

import torch.nn as nn

class ActivityClassifier(nn.Module):
    def __init__(self, embedding_dim=128, num_classes=10):
        super(ActivityClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)


# ============================================
# CONFIGURATION
# ============================================

BASE_PATH = r"C:\Users\Amrit Shah\Desktop\protobyte 2.0\protobytes-2.0-team-Coder-Rex"
MODELS_PATH = os.path.join(BASE_PATH, "models")
DEMO_PATH = os.path.join(BASE_PATH, "demo_samples")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EMOJI_MAP = {
    'Walking': '🚶', 'Running': '🏃', 'Sitting': '🪑',
    'Standing': '🧍', 'Lying': '🛏️', 'Fall': '🚨',
    'Jumping': '🤸', 'Cycling': '🚴', 'Stairs_Up': '⬆️',
    'Stairs_Down': '⬇️', 'Other': '❓'
}


# ============================================
# LOAD MODELS (Cached)
# ============================================

@st.cache_resource
def load_models():
    """Load all models once"""
    
    encoder = IMUEncoder(input_channels=3, embedding_dim=128)
    encoder.load_state_dict(torch.load(
        os.path.join(MODELS_PATH, "encoder.pth"), map_location=DEVICE
    ))
    encoder = encoder.to(DEVICE)
    encoder.eval()

    with open(os.path.join(MODELS_PATH, "label_map.json"), 'r') as f:
        label_map = json.load(f)

    num_classes = len(label_map)
    classifier = ActivityClassifier(embedding_dim=128, num_classes=num_classes)
    classifier.load_state_dict(torch.load(
        os.path.join(MODELS_PATH, "multiclass_classifier.pth"), map_location=DEVICE
    ))
    classifier = classifier.to(DEVICE)
    classifier.eval()

    return encoder, classifier, label_map


# ============================================
# PREDICTION FUNCTION
# ============================================

def predict(raw_data, encoder, classifier, label_map, input_hz=50):
    """
    Input: raw numpy array (N, 3) — accel_x, accel_y, accel_z
    Output: prediction dict
    """
    
    # Resample if needed
    if input_hz != 50:
        target_len = int(len(raw_data) * 50 / input_hz)
        resampled = np.zeros((target_len, 3))
        for col in range(3):
            resampled[:, col] = signal.resample(raw_data[:, col], target_len)
        raw_data = resampled

    # Create windows
    windows = []
    for start in range(0, len(raw_data) - 128, 64):
        window = raw_data[start:start + 128]
        scaler = StandardScaler()
        window = scaler.fit_transform(window)
        windows.append(window)

    if not windows:
        return None

    windows = np.array(windows, dtype=np.float32)

    # Predict
    with torch.no_grad():
        tensor = torch.tensor(windows, dtype=torch.float32).to(DEVICE)
        emb = encoder.get_embedding(tensor)
        out = classifier(emb)
        probs = torch.softmax(out, dim=1)
        preds = torch.argmax(out, dim=1)

    # Results
    avg_probs = probs.mean(dim=0).cpu().numpy()
    all_predictions = {}
    for idx_str, name in label_map.items():
        all_predictions[name] = round(float(avg_probs[int(idx_str)]) * 100, 2)

    all_predictions = dict(sorted(all_predictions.items(), key=lambda x: -x[1]))

    top_idx = np.argmax(avg_probs)
    top_activity = label_map[str(top_idx)]
    top_confidence = float(avg_probs[top_idx]) * 100

    # Per window
    window_results = []
    for i in range(len(windows)):
        act = label_map[str(preds[i].item())]
        conf = probs[i][preds[i]].item() * 100
        window_results.append({'activity': act, 'confidence': conf})

    return {
        'activity': top_activity,
        'confidence': top_confidence,
        'all_predictions': all_predictions,
        'is_fall': top_activity == 'Fall',
        'per_window': window_results,
        'num_windows': len(windows),
        'embedding': emb[0].cpu().numpy()
    }


# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="Universal Activity Detection",
    page_icon="🩺",
    layout="wide"
)

# Load models
encoder, classifier, label_map = load_models()

# ============================================
# SIDEBAR
# ============================================

st.sidebar.title("🩺 About")
st.sidebar.info("""
**Universal Activity Detection System**

Detects human activities from ANY wearable device 
using self-supervised contrastive learning.

**Activities:**
🚶 Walking | 🏃 Running | 🪑 Sitting
🧍 Standing | 🛏️ Lying | 🚨 Fall
🤸 Jumping | 🚴 Cycling | ⬆️ Stairs Up
⬇️ Stairs Down

**Accuracy: 98.3%**
**Devices: Any IMU sensor**
""")

# ============================================
# MAIN TITLE
# ============================================

st.title("🩺 Universal Activity Detection System")
st.markdown("### Detect activities from ANY wearable device")

# ============================================
# TABS
# ============================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📁 Upload CSV",
    "📱 Live from Phone",
    "🧪 Test Samples",
    "ℹ️ How It Works"
])

# ============================================
# TAB 1: UPLOAD CSV
# ============================================

with tab1:
    st.header("📁 Upload Sensor Data")

    col1, col2 = st.columns(2)

    with col1:
        uploaded_file = st.file_uploader(
            "Upload CSV file with accelerometer data",
            type=['csv'],
            help="CSV with columns: accel_x, accel_y, accel_z (or any 3 numeric columns)"
        )

        input_hz = st.number_input(
            "Sampling Rate (Hz)", 
            min_value=10, max_value=500, value=50,
            help="How many readings per second. Phyphox = ~100Hz, Smartwatch = ~50Hz"
        )

    with col2:
        st.markdown("**Expected CSV Format:**")
        st.code("""accel_x,accel_y,accel_z
0.52,0.31,9.81
0.54,0.33,9.79
0.56,0.35,9.77
...(128+ rows needed)""")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        
        st.success(f"✅ Loaded {len(data)} rows")

        # Find numeric columns
        if 'accel_x' in data.columns:
            raw = data[['accel_x', 'accel_y', 'accel_z']].values
        else:
            numeric = data.select_dtypes(include=[np.number])
            raw = numeric.iloc[:, :3].values
            st.info(f"Using columns: {list(numeric.columns[:3])}")

        # Show raw data
        with st.expander("View Raw Data"):
            st.dataframe(data.head(20))

            fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                               subplot_titles=('X-axis', 'Y-axis', 'Z-axis'))
            fig.add_trace(go.Scatter(y=raw[:500, 0], name='X', line=dict(color='red')), row=1, col=1)
            fig.add_trace(go.Scatter(y=raw[:500, 1], name='Y', line=dict(color='green')), row=2, col=1)
            fig.add_trace(go.Scatter(y=raw[:500, 2], name='Z', line=dict(color='blue')), row=3, col=1)
            fig.update_layout(height=500, title_text="Raw Sensor Data")
            st.plotly_chart(fig, use_container_width=True)

        # Predict button
        if st.button("🔍 Detect Activity", type="primary", key="csv_predict"):
            with st.spinner("Analyzing..."):
                result = predict(raw, encoder, classifier, label_map, input_hz)

            if result is None:
                st.error("❌ Not enough data! Need at least 128 samples (2.56 seconds at 50Hz)")
            else:
                # Show result
                emoji = EMOJI_MAP.get(result['activity'], '📊')

                if result['is_fall']:
                    st.error(f"## 🚨 FALL DETECTED! Confidence: {result['confidence']:.1f}%")
                else:
                    st.success(f"## {emoji} {result['activity']} Detected! Confidence: {result['confidence']:.1f}%")

                # Metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Activity", result['activity'])
                col2.metric("Confidence", f"{result['confidence']:.1f}%")
                col3.metric("Windows Analyzed", result['num_windows'])

                # All predictions bar chart
                st.subheader("All Predictions")
                pred_df = pd.DataFrame({
                    'Activity': list(result['all_predictions'].keys()),
                    'Confidence': list(result['all_predictions'].values())
                })

                fig = go.Figure(data=[
                    go.Bar(
                        x=pred_df['Activity'],
                        y=pred_df['Confidence'],
                        marker_color=['red' if a == 'Fall' else 'steelblue' 
                                     for a in pred_df['Activity']]
                    )
                ])
                fig.update_layout(
                    title="Activity Probabilities",
                    xaxis_title="Activity",
                    yaxis_title="Confidence (%)",
                    yaxis_range=[0, 100]
                )
                st.plotly_chart(fig, use_container_width=True)

                # Per window results
                with st.expander(f"Per-Window Analysis ({result['num_windows']} windows)"):
                    for i, wr in enumerate(result['per_window']):
                        e = EMOJI_MAP.get(wr['activity'], '📊')
                        st.write(f"Window {i+1}: {e} {wr['activity']} ({wr['confidence']:.1f}%)")


# ============================================
# TAB 2: LIVE FROM PHONE (PHYPHOX) - FIXED
# ============================================

with tab2:
    st.header("📱 Live Detection from Phone")

    st.markdown("""
    ### Setup Instructions:
    1. 📱 Install **Phyphox** app on your phone
    2. Open **"Accelerometer (without g)"** or **"Acceleration with g"**
    3. Tap **⋮** (three dots) → **"Allow Remote Access"**
    4. Note the **IP address** shown
    5. Make sure phone and laptop are on **SAME WiFi**
    6. Press **▶ PLAY** in Phyphox
    7. Enter IP below and click **"Start Live Detection"**
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        phone_ip = st.text_input(
            "Phyphox IP Address",
            value="192.168.4.227",
            help="The IP shown in Phyphox when you enable Remote Access"
        )

    with col2:
        phone_port = st.text_input(
            "Port",
            value="8080",
            help="Default Phyphox port is 8080"
        )

    with col3:
        phone_hz = st.number_input(
            "Phone Sample Rate (Hz)",
            min_value=10, max_value=500, value=200,
            help="Your phone records at ~200Hz (shown when you collected data)"
        )

    phyphox_url = f"http://{phone_ip}:{phone_port}"

    # Test connection
    if st.button("🔗 Test Connection"):
        try:
            response = requests.get(
                f"{phyphox_url}/get?accX=full",
                timeout=3
            )
            if response.status_code == 200:
                st.success(f"✅ Connected to phone at {phyphox_url}")
            else:
                st.error(f"❌ Got status {response.status_code}")
        except:
            st.error("❌ Cannot connect! Check IP, WiFi, and Phyphox Remote Access")

    st.markdown("---")

    # Record duration
    duration = st.slider("Recording Duration (seconds)", 3, 15, 5)

    # Activity selector for what you're about to do
    activity_doing = st.selectbox(
        "What activity will you perform?",
        ["Walking", "Running", "Sitting", "Standing", "Jumping",
         "Stairs Up", "Stairs Down", "Fall (careful!)", "Other"]
    )

    if st.button("🎯 Start Live Detection", type="primary"):

        progress = st.progress(0)
        status = st.empty()

        # ===== STEP 1: CLEAR OLD DATA =====
        status.info("🗑️ Clearing old data...")
        try:
            requests.get(f"{phyphox_url}/control?cmd=clear", timeout=3)
            time.sleep(0.5)
        except:
            pass

        # ===== STEP 2: START RECORDING =====
        status.info("▶️ Starting recording...")
        try:
            requests.get(f"{phyphox_url}/control?cmd=start", timeout=3)
        except:
            pass

        # ===== STEP 3: WAIT FOR DATA =====
        status.warning(
            f"📱 Recording for {duration} seconds... "
            f"Perform **{activity_doing}** now!"
        )

        for i in range(duration):
            time.sleep(1)
            elapsed = i + 1
            progress.progress(elapsed / duration)
            status.warning(
                f"⏱️ {duration - elapsed} seconds remaining... "
                f"Keep doing **{activity_doing}**!"
            )

        progress.progress(1.0)

        # ===== STEP 4: FETCH ALL DATA AT ONCE =====
        status.info("📥 Fetching all recorded data...")

        try:
            response = requests.get(
                f"{phyphox_url}/get?"
                f"accX=full&accY=full&accZ=full&acc_time=full",
                timeout=10
            )
            data = response.json()

            x_raw = data['buffer']['accX']['buffer']
            y_raw = data['buffer']['accY']['buffer']
            z_raw = data['buffer']['accZ']['buffer']

            # Remove None values
            all_x, all_y, all_z = [], [], []
            for i in range(len(x_raw)):
                if (x_raw[i] is not None and
                    y_raw[i] is not None and
                    z_raw[i] is not None):
                    all_x.append(x_raw[i])
                    all_y.append(y_raw[i])
                    all_z.append(z_raw[i])

        except Exception as e:
            st.error(f"❌ Failed to fetch data: {e}")
            all_x, all_y, all_z = [], [], []

        # ===== STEP 5: ANALYZE =====
        if len(all_x) < 128:
            st.error(
                f"❌ Not enough data! Got {len(all_x)} readings, need 128+\n\n"
                f"Try recording for longer or check connection"
            )
        else:
            status.success(
                f"✅ Recorded {len(all_x)} readings! "
                f"(~{len(all_x)/duration:.0f} Hz)"
            )

            # Convert to numpy
            raw = np.column_stack([all_x, all_y, all_z])

            # Show raw data
            with st.expander("📊 View Raw Phone Data"):
                fig = make_subplots(
                    rows=3, cols=1, shared_xaxes=True,
                    subplot_titles=('X-axis', 'Y-axis', 'Z-axis')
                )
                # Only show first 1000 points for speed
                show_n = min(1000, len(all_x))
                fig.add_trace(go.Scatter(
                    y=all_x[:show_n], name='X',
                    line=dict(color='red')
                ), row=1, col=1)
                fig.add_trace(go.Scatter(
                    y=all_y[:show_n], name='Y',
                    line=dict(color='green')
                ), row=2, col=1)
                fig.add_trace(go.Scatter(
                    y=all_z[:show_n], name='Z',
                    line=dict(color='blue')
                ), row=3, col=1)
                fig.update_layout(
                    height=500,
                    title_text=f"Phone Sensor Data ({len(all_x)} readings)"
                )
                st.plotly_chart(fig, use_container_width=True)

            # ===== PREDICT =====
            with st.spinner("🧠 AI is analyzing your motion..."):
                result = predict(
                    raw, encoder, classifier, label_map,
                    input_hz=phone_hz  # Use actual phone Hz!
                )

            if result:
                emoji = EMOJI_MAP.get(result['activity'], '📊')

                # Big result display
                if result['is_fall']:
                    st.error(
                        f"## 🚨 FALL DETECTED!\n"
                        f"### Confidence: {result['confidence']:.1f}%"
                    )
                else:
                    st.success(
                        f"## {emoji} {result['activity']} Detected!\n"
                        f"### Confidence: {result['confidence']:.1f}%"
                    )

                # Metrics row
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Activity", result['activity'])
                col2.metric("Confidence", f"{result['confidence']:.1f}%")
                col3.metric("Data Points", len(all_x))
                col4.metric("Windows", result['num_windows'])

                # All predictions bar chart
                st.subheader("All Activity Probabilities")
                pred_df = pd.DataFrame({
                    'Activity': list(result['all_predictions'].keys()),
                    'Confidence': list(result['all_predictions'].values())
                })

                fig = go.Figure(data=[
                    go.Bar(
                        x=pred_df['Activity'],
                        y=pred_df['Confidence'],
                        marker_color=[
                            'red' if a == 'Fall'
                            else 'green' if a == result['activity']
                            else 'steelblue'
                            for a in pred_df['Activity']
                        ],
                        text=[f"{c:.1f}%" for c in pred_df['Confidence']],
                        textposition='auto'
                    )
                ])
                fig.update_layout(
                    title="Activity Probabilities",
                    xaxis_title="Activity",
                    yaxis_title="Confidence (%)",
                    yaxis_range=[0, 100],
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

                # Per window timeline
                with st.expander(
                    f"📋 Per-Window Analysis ({result['num_windows']} windows)"
                ):
                    for i, wr in enumerate(result['per_window']):
                        e = EMOJI_MAP.get(wr['activity'], '📊')
                        bar_len = int(wr['confidence'] / 5)
                        bar = "█" * bar_len
                        st.write(
                            f"Window {i+1}: {e} **{wr['activity']}** "
                            f"({wr['confidence']:.1f}%) {bar}"
                        )

    # ===== ALTERNATIVE: Upload CSV from Phyphox =====
    st.markdown("---")
    st.subheader("📎 Alternative: Upload Phyphox CSV Export")
    st.markdown(
        "If live connection doesn't work, export CSV from Phyphox "
        "and upload here:"
    )

    phyphox_file = st.file_uploader(
        "Upload Phyphox CSV", type=['csv'], key="phyphox_csv"
    )

    if phyphox_file is not None:
        data = pd.read_csv(phyphox_file)
        st.write(f"Columns: {list(data.columns)}")

        numeric = data.select_dtypes(include=[np.number])

        if len(numeric.columns) >= 3:
            # Try to detect column names
            col_names = list(numeric.columns)
            st.info(f"Using columns: {col_names[:3]}")

            raw = numeric.iloc[:, :3].values
            st.success(f"✅ Loaded {len(raw)} readings")

            if st.button("🔍 Analyze Phyphox Data", key="phyphox_analyze"):
                result = predict(
                    raw, encoder, classifier, label_map,
                    input_hz=phone_hz
                )

                if result:
                    emoji = EMOJI_MAP.get(result['activity'], '📊')
                    st.success(
                        f"## {emoji} {result['activity']} "
                        f"({result['confidence']:.1f}%)"
                    )

# ============================================
# TAB 3: TEST SAMPLES
# ============================================

with tab3:
    st.header("🧪 Test with Pre-made Samples")

    if os.path.exists(DEMO_PATH):
        files = sorted([f for f in os.listdir(DEMO_PATH) if f.endswith('.csv')])

        if files:
            selected_file = st.selectbox("Select a test file", files)

            col1, col2 = st.columns(2)

            with col1:
                # Parse expected activity from filename
                parts = selected_file.replace('test_', '').replace('.csv', '').split('_')
                expected = parts[0].capitalize()
                source = parts[1].upper() if len(parts) > 1 else "Unknown"

                st.info(f"**Expected Activity:** {expected}")
                st.info(f"**Source Device:** {source}")

            with col2:
                if st.button("🔍 Analyze", type="primary", key="test_analyze"):
                    filepath = os.path.join(DEMO_PATH, selected_file)
                    data = pd.read_csv(filepath)
                    raw = data[['accel_x', 'accel_y', 'accel_z']].values

                    result = predict(raw, encoder, classifier, label_map)

                    if result:
                        emoji = EMOJI_MAP.get(result['activity'], '📊')
                        
                        if result['activity'].lower() == expected.lower():
                            st.success(f"## {emoji} {result['activity']} ({result['confidence']:.1f}%) ✅ Correct!")
                        else:
                            st.warning(f"## {emoji} {result['activity']} ({result['confidence']:.1f}%) (Expected: {expected})")

            # Test ALL button
            st.markdown("---")
            if st.button("🧪 Test ALL Files", key="test_all"):
                results_table = []
                correct = 0
                total = 0

                progress = st.progress(0)

                for i, filename in enumerate(files):
                    filepath = os.path.join(DEMO_PATH, filename)
                    data = pd.read_csv(filepath)
                    raw = data[['accel_x', 'accel_y', 'accel_z']].values

                    result = predict(raw, encoder, classifier, label_map)

                    parts = filename.replace('test_', '').replace('.csv', '').split('_')
                    expected_act = parts[0].capitalize()

                    if result:
                        detected = result['activity']
                        conf = result['confidence']
                        match = expected_act.lower() in detected.lower() or detected.lower() in expected_act.lower()

                        total += 1
                        if match:
                            correct += 1

                        results_table.append({
                            'File': filename,
                            'Expected': expected_act,
                            'Detected': detected,
                            'Confidence': f"{conf:.1f}%",
                            'Correct': '✅' if match else '❌'
                        })

                    progress.progress((i + 1) / len(files))

                st.dataframe(pd.DataFrame(results_table), use_container_width=True)
                
                if total > 0:
                    acc = 100 * correct / total
                    st.metric("Demo Accuracy", f"{acc:.1f}% ({correct}/{total})")
        else:
            st.warning("No test files found! Run training first.")
    else:
        st.warning(f"Demo folder not found: {DEMO_PATH}")


# ============================================
# TAB 4: HOW IT WORKS
# ============================================

with tab4:
    st.header("ℹ️ How It Works")

    st.markdown("""
    ## The Problem
    
    Different wearable devices (Apple Watch, Fitbit, Samsung) record motion data 
    in different formats. **One health app can't work with all devices.**
    
    ## Our Solution
    
    We use **Self-Supervised Contrastive Learning** to create 
    **device-agnostic motion embeddings**.
    
    ## Pipeline
    
    ```
    📱 Any Wearable Device
         ↓
    📊 Raw X, Y, Z Data
         ↓
    🔄 Preprocessing (Resample, Normalize, Window)
         ↓
    🧠 1D-CNN Encoder → 128-dim Embedding
         ↓
    🎯 Activity Classifier → Walking/Running/Fall/etc.
    ```
    
    ## Training Data
    
    | Dataset | Type | Activities | Accuracy |
    |---------|------|------------|----------|
    | SisFall | Medical Sensor | Falls + Daily | 97.6% |
    | WISDM | Smartwatch | Walking, Running, etc. | 100% |
    | PAMAP2 | Research Sensor | 12 activities | 96.0% |
    
    ## Key Stats
    
    - **Total samples:** 112,677
    - **Activities detected:** 10+
    - **Overall accuracy:** 98.3%
    - **Model size:** 69,280 parameters
    - **Inference time:** <100ms
    """)

# ============================================
# FOOTER
# ============================================

st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built for Protobyte Hackathon | Team: Coder Rex</p>
    <p>Self-Supervised Contrastive Learning for Cross-Device Wearable Standardization</p>
</div>
""", unsafe_allow_html=True)