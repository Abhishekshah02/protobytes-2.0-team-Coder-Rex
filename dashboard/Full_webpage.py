import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from io import StringIO
import time
import uuid
import pytz

# ML imports
import torch
import torch.nn as nn
import json
import os
import sys
from scipy import signal
from sklearn.preprocessing import StandardScaler

# Add your model path - ADJUST THIS PATH
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from model import IMUEncoder

# ============================================
# ML MODEL CLASSES
# ============================================


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
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.classifier(x)


# ============================================
# ML CONFIGURATION
# ============================================

BASE_PATH = r"C:\Users\Amrit Shah\Desktop\protobyte 2.0\protobytes-2.0-team-Coder-Rex\models"  # ADJUST THIS PATH
MODELS_PATH = os.path.join(BASE_PATH)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMOJI_MAP = {
    "Walking": "🚶",
    "Running": "🏃",
    "Sitting": "🪑",
    "Standing": "🧍",
    "Lying": "🛏️",
    "Fall": "🚨",
    "Jumping": "🤸",
    "Cycling": "🚴",
    "Stairs_Up": "⬆️",
    "Stairs_Down": "⬇️",
    "Other": "❓",
}

# ============================================
# LOAD ML MODELS (Cached)
# ============================================


@st.cache_resource
def load_models():
    """Load ML models once - CRITICAL: Must be cached!"""
    try:
        encoder = IMUEncoder(input_channels=3, embedding_dim=128)
        encoder.load_state_dict(
            torch.load(os.path.join(MODELS_PATH, "encoder.pth"), map_location=DEVICE)
        )
        encoder = encoder.to(DEVICE)
        encoder.eval()

        with open(os.path.join(MODELS_PATH, "label_map.json"), "r") as f:
            label_map = json.load(f)

        num_classes = len(label_map)
        classifier = ActivityClassifier(embedding_dim=128, num_classes=num_classes)
        classifier.load_state_dict(
            torch.load(
                os.path.join(MODELS_PATH, "multiclass_classifier.pth"),
                map_location=DEVICE,
            )
        )
        classifier = classifier.to(DEVICE)
        classifier.eval()

        return encoder, classifier, label_map
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        return None, None, None


# ============================================
# PREDICTION FUNCTION (EXACT COPY from analyzer.py)
# ============================================


def predict(raw_data, encoder, classifier, label_map, input_hz=50):
    """
    CRITICAL: Must match analyzer.py exactly
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

    # Create windows - CRITICAL: Each window normalized independently
    windows = []
    for start in range(0, len(raw_data) - 128, 64):
        window = raw_data[start : start + 128]
        scaler = StandardScaler()
        window = scaler.fit_transform(window)  # FIT on each window!
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
        window_results.append({"activity": act, "confidence": conf})

    return {
        "activity": top_activity,
        "confidence": top_confidence,
        "all_predictions": all_predictions,
        "is_fall": top_activity == "Fall",
        "per_window": window_results,
        "num_windows": len(windows),
        "embedding": emb[0].cpu().numpy(),
    }


# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="Clinical Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS (same as before)
st.markdown(
    """
<style>
    .stApp {
        background-color: #0f1419;
    }
    .main-header {
        font-size: 2rem;
        font-weight: 600;
        color: white;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: #9ca3af;
        font-size: 0.9rem;
    }
    .alert-card-red {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(127, 29, 29, 0.2));
        border: 1px solid rgba(239, 68, 68, 0.3);
        padding: 1rem;
        border-radius: 0.75rem;
    }
    .alert-card-orange {
        background: linear-gradient(135deg, rgba(251, 146, 60, 0.2), rgba(124, 45, 18, 0.2));
        border: 1px solid rgba(251, 146, 60, 0.3);
        padding: 1rem;
        border-radius: 0.75rem;
    }
    .alert-card-yellow {
        background: linear-gradient(135deg, rgba(250, 204, 21, 0.2), rgba(120, 53, 15, 0.2));
        border: 1px solid rgba(250, 204, 21, 0.3);
        padding: 1rem;
        border-radius: 0.75rem;
    }
    .patient-divider {
        border: none;
        border-top: 1px solid #2d3748;
        margin: 8px 0;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ========================
# Session State
# ========================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = "Hackathon"
if "current_page" not in st.session_state:
    st.session_state.current_page = "Dashboard"
if "patients" not in st.session_state:
    st.session_state.patients = []
if "patient_counter" not in st.session_state:
    st.session_state.patient_counter = 10000
if "show_add_patient_form" not in st.session_state:
    st.session_state.show_add_patient_form = False
if "uploaded_file_data" not in st.session_state:
    st.session_state.uploaded_file_data = None
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

# ========================
# Helper functions (keep all your existing ones)
# ========================


def get_greeting():
    kathmandu_tz = pytz.timezone("Asia/Kathmandu")
    now = datetime.now(kathmandu_tz)
    hour = now.hour
    if 5 <= hour < 12:
        return "Good morning"
    elif 12 <= hour < 17:
        return "Good afternoon"
    elif 17 <= hour < 21:
        return "Good evening"
    else:
        return "Good night"


def generate_patient_id():
    st.session_state.patient_counter += 1
    return f"#{st.session_state.patient_counter}"


def add_patient(name, problem, age, sex, notes=""):
    patient_id = generate_patient_id()
    unique_key = uuid.uuid4().hex[:8]
    new_patient = {
        "id": patient_id,
        "unique_key": unique_key,
        "name": name,
        "condition": problem,
        "age": age,
        "gender": sex,
        "notes": notes,
        "last_visit": "New Patient",
        "next_session": "Not Scheduled",
        "status": "active",
        "created_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    st.session_state.patients.append(new_patient)
    return patient_id


def delete_patient(patient_id):
    st.session_state.patients = [
        p for p in st.session_state.patients if p["id"] != patient_id
    ]


def get_session_requests():
    return []


# ========================
# Login page (keep as is)
# ========================
def login_page():
    st.markdown(
        '<h1 style="text-align: center; color: white;">Sign in to your account</h1>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        with st.form("login_form"):
            username = st.text_input("Doctor ID", placeholder="Enter your Doctor ID")
            password = st.text_input(
                "Password", type="password", placeholder="Enter your password"
            )
            remember = st.checkbox("Remember me")

            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                submitted = st.form_submit_button(
                    "Sign in", use_container_width=True, type="primary"
                )
            with col_btn2:
                if st.form_submit_button("Forgot password?", use_container_width=True):
                    st.info("Password recovery feature coming soon!")

            if submitted:
                if username == "admin" and password == "admin123":
                    st.session_state.logged_in = True
                    st.session_state.username = "Hackathon"
                    st.rerun()
                else:
                    st.error("Invalid Doctor ID or Password. Please try again.")

        st.markdown("---")
        st.markdown(
            '<p style="text-align: center; color: #9ca3af;">Not a member? <a href="#" style="color: #667EEA;">Register as doctor</a></p>',
            unsafe_allow_html=True,
        )


# ========================
# Dashboard page (keep as is)
# ========================
def dashboard_page():
    greeting = get_greeting()
    st.markdown(
        f'<h1 class="main-header">{greeting}, {st.session_state.username}</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="sub-header">Here is your clinical overview for today.</p>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
        <div class="alert-card-red">
            <p style="color: #fca5a5; font-size: 0.9rem; margin-bottom: 0.5rem;">Fall Alerts Today</p>
            <h2 style="color: #f87171; font-size: 2.5rem; margin: 0;">0</h2>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
        <div class="alert-card-orange">
            <p style="color: #fed7aa; font-size: 0.9rem; margin-bottom: 0.5rem;">Seizure Alerts Today</p>
            <h2 style="color: #fb923c; font-size: 2.5rem; margin: 0;">0</h2>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
        <div class="alert-card-yellow">
            <p style="color: #fef3c7; font-size: 0.9rem; margin-bottom: 0.5rem;">Other Emergency</p>
            <h2 style="color: #fbbf24; font-size: 2.5rem; margin: 0;">0</h2>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("Recent Patients")
        patients = st.session_state.patients[-5:] if st.session_state.patients else []

        if patients:
            for patient in reversed(patients):
                col_p1, col_p2, col_p3 = st.columns([3, 2, 1])
                with col_p1:
                    st.markdown(f"**{patient['name']}**")
                    st.caption(f"{patient['condition']} | {patient['id']}")
                with col_p2:
                    st.caption(f"Last visit: {patient['last_visit']}")
                with col_p3:
                    if patient["status"] == "active":
                        st.markdown(
                            '<span style="color: #10b981;">Active</span>',
                            unsafe_allow_html=True,
                        )
                st.markdown("---")
        else:
            st.info("No patients added yet. Go to Patients page to add patients.")

        st.subheader("Session Requests")
        requests = get_session_requests()
        if not requests:
            st.info("No session requests at this time.")

    with col_right:
        total_patients = len(st.session_state.patients)
        st.metric("My Clients", total_patients, "0 sessions this week")


# ========================
# Add Patient Form
# ========================
def add_patient_form():
    st.subheader("Add New Patient")

    with st.form("add_patient_form", clear_on_submit=True):
        col1, col2 = st.columns(2)

        with col1:
            name = st.text_input("Patient Name *", placeholder="Enter full name")
            age = st.number_input("Age *", min_value=0, max_value=150, value=30)

        with col2:
            problem = st.text_input(
                "Problem / Condition *", placeholder="e.g., Anxiety, Depression"
            )
            sex = st.selectbox("Sex *", ["Male", "Female", "Other"])

        notes = st.text_area(
            "Additional Notes", placeholder="Any additional information..."
        )

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            submitted = st.form_submit_button(
                "Add Patient", use_container_width=True, type="primary"
            )
        with col_btn2:
            cancel = st.form_submit_button("Cancel", use_container_width=True)

        if submitted:
            if name and problem and age:
                patient_id = add_patient(name, problem, age, sex, notes)
                st.success(f"Patient added successfully! Patient ID: {patient_id}")
                st.session_state.show_add_patient_form = False
                time.sleep(1)
                st.rerun()
            else:
                st.error("Please fill in all required fields (Name, Problem, Age)")

        if cancel:
            st.session_state.show_add_patient_form = False
            st.rerun()


# ========================
# Add Patient Form (Simplified - Connection Required)
# ========================
def add_patient_form():
    st.subheader("Add New Patient")

    # Initialize connection status in session state
    if "phone_connected" not in st.session_state:
        st.session_state.phone_connected = False

    # ============================================
    # STEP 1: PHYPHOX CONNECTION (REQUIRED FIRST)
    # ============================================

    col1, col2, col3 = st.columns(3)

    with col1:
        phone_ip = st.text_input(
            "Phyphox IP Address",
            value="192.168.4.227",
            help="The IP shown in Phyphox when you enable Remote Access",
            key="patient_phone_ip",
        )

    with col2:
        phone_port = st.text_input(
            "Port",
            value="8080",
            help="Default Phyphox port is 8080",
            key="patient_phone_port",
        )

    with col3:
        phone_hz = st.number_input(
            "Phone Sample Rate (Hz)",
            min_value=10,
            max_value=500,
            value=200,
            help="Your phone records at ~200Hz",
            key="patient_phone_hz",
        )

    phyphox_url = f"http://{phone_ip}:{phone_port}"

    # Test connection button
    col_test1, col_test2 = st.columns([1, 3])
    with col_test1:
        if st.button(
            "Test Connection", key="patient_test_conn", width="stretch", type="primary"
        ):
            try:
                import requests

                response = requests.get(f"{phyphox_url}/get?accX=full", timeout=3)
                if response.status_code == 200:
                    st.session_state.phone_connected = True
                    st.success(f"Connected to phone at {phyphox_url}")
                else:
                    st.session_state.phone_connected = False
                    st.error(f"Got status {response.status_code}")
            except Exception as e:
                st.session_state.phone_connected = False
                st.error(f"Cannot connect! Check IP, WiFi, and Phyphox Remote Access")

    with col_test2:
        if st.session_state.phone_connected:
            st.success("Phone connected! You can now add the patient.")
        else:
            st.warning("Please test connection first before adding patient.")

    st.markdown("---")

    # ============================================
    # STEP 2: PATIENT FORM (ONLY IF CONNECTED)
    # ============================================

    if not st.session_state.phone_connected:
        st.info("Please connect to phone first to enable patient registration.")

        # Show cancel button even when not connected
        if st.button("Cancel", width="stretch", key="cancel_no_conn"):
            st.session_state.show_add_patient_form = False
            st.session_state.phone_connected = False
            st.rerun()
        return

    st.markdown("### Step 2: Enter Patient Details")

    with st.form("add_patient_form", clear_on_submit=True):
        col1, col2 = st.columns(2)

        with col1:
            name = st.text_input("Patient Name *", placeholder="Enter full name")
            age = st.number_input("Age *", min_value=0, max_value=150, value=30)

        with col2:
            problem = st.text_input(
                "Problem / Condition *", placeholder="e.g., Anxiety, Depression"
            )
            sex = st.selectbox("Sex *", ["Male", "Female", "Other"])

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            submitted = st.form_submit_button(
                "Add Patient", width="stretch", type="primary"
            )
        with col_btn2:
            cancel = st.form_submit_button("Cancel", width="stretch")

        if submitted:
            if name and problem and age:
                patient_id = add_patient(name, problem, age, sex, notes="")
                st.success(f"Patient added successfully! Patient ID: {patient_id}")
                st.info(
                    "Click 'Details' button on the patient to run live activity detection."
                )
                st.session_state.show_add_patient_form = False
                st.session_state.phone_connected = False  # Reset connection status
                time.sleep(2)
                st.rerun()
            else:
                st.error("Please fill in all required fields (Name, Problem, Age)")

        if cancel:
            st.session_state.show_add_patient_form = False
            st.session_state.phone_connected = False  # Reset connection status
            st.rerun()


# ========================
# Patient Details Modal with Live Detection
# ========================
@st.dialog("Patient Details & Live Activity Detection", width="large")
def patient_details_modal(patient):
    """Modal showing patient details and live detection capability"""

    st.markdown(f"### {patient['name']}")

    # Patient info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Patient ID", patient["id"])
        st.metric("Age", patient["age"])
    with col2:
        st.metric("Gender", patient["gender"])
        st.metric("Condition", patient["condition"])
    with col3:
        st.metric("Status", patient["status"].title())
        st.metric("Last Visit", patient["last_visit"])

    st.markdown("---")

    # ============================================
    # LIVE ACTIVITY DETECTION SECTION
    # ============================================

    st.markdown("### Live Activity Detection")

    col1, col2, col3 = st.columns(3)

    with col1:
        phone_ip = st.text_input(
            "Phyphox IP Address",
            value="192.168.4.227",
            help="The IP shown in Phyphox when you enable Remote Access",
            key=f"detail_phone_ip_{patient['id']}",
        )

    with col2:
        phone_port = st.text_input(
            "Port",
            value="8080",
            help="Default Phyphox port is 8080",
            key=f"detail_phone_port_{patient['id']}",
        )

    with col3:
        phone_hz = st.number_input(
            "Phone Sample Rate (Hz)",
            min_value=10,
            max_value=500,
            value=200,
            help="Your phone records at ~200Hz",
            key=f"detail_phone_hz_{patient['id']}",
        )

    phyphox_url = f"http://{phone_ip}:{phone_port}"

    # Test connection
    if st.button("Test Connection", key=f"detail_test_conn_{patient['id']}"):
        try:
            import requests

            response = requests.get(f"{phyphox_url}/get?accX=full", timeout=3)
            if response.status_code == 200:
                st.success(f"Connected to phone at {phyphox_url}")
            else:
                st.error(f"Got status {response.status_code}")
        except Exception as e:
            st.error(f"Cannot connect! Check IP, WiFi, and Phyphox Remote Access")

    st.markdown("---")

    # Record duration
    duration = st.slider(
        "Recording Duration (seconds)", 3, 15, 5, key=f"detail_duration_{patient['id']}"
    )

    # Activity selector
    activity_doing = st.selectbox(
        "What activity will the patient perform?",
        [
            "Walking",
            "Running",
            "Sitting",
            "Standing",
            "Jumping",
            "Stairs Up",
            "Stairs Down",
            "Fall (careful!)",
            "Other",
        ],
        key=f"detail_activity_{patient['id']}",
    )

    if st.button(
        "Start Live Detection",
        type="primary",
        key=f"detail_live_detect_{patient['id']}",
    ):

        # Load models first
        encoder, classifier, label_map = load_models()

        if encoder is None:
            st.error("ML models not loaded! Check model paths.")
            return

        import requests

        progress = st.progress(0)
        status = st.empty()

        # ===== STEP 1: CLEAR OLD DATA =====
        status.info("Clearing old data...")
        try:
            requests.get(f"{phyphox_url}/control?cmd=clear", timeout=3)
            time.sleep(0.5)
        except:
            pass

        # ===== STEP 2: START RECORDING =====
        status.info("Starting recording...")
        try:
            requests.get(f"{phyphox_url}/control?cmd=start", timeout=3)
        except:
            pass

        # ===== STEP 3: WAIT FOR DATA =====
        status.warning(
            f"Recording for {duration} seconds... " f"Perform **{activity_doing}** now!"
        )

        for i in range(duration):
            time.sleep(1)
            elapsed = i + 1
            progress.progress(elapsed / duration)
            status.warning(
                f"{duration - elapsed} seconds remaining... "
                f"Keep doing **{activity_doing}**!"
            )

        progress.progress(1.0)

        # ===== STEP 4: FETCH ALL DATA AT ONCE =====
        status.info("Fetching all recorded data...")

        try:
            response = requests.get(
                f"{phyphox_url}/get?" f"accX=full&accY=full&accZ=full&acc_time=full",
                timeout=10,
            )
            data = response.json()

            x_raw = data["buffer"]["accX"]["buffer"]
            y_raw = data["buffer"]["accY"]["buffer"]
            z_raw = data["buffer"]["accZ"]["buffer"]

            # Remove None values
            all_x, all_y, all_z = [], [], []
            for i in range(len(x_raw)):
                if (
                    x_raw[i] is not None
                    and y_raw[i] is not None
                    and z_raw[i] is not None
                ):
                    all_x.append(x_raw[i])
                    all_y.append(y_raw[i])
                    all_z.append(z_raw[i])

        except Exception as e:
            st.error(f"Failed to fetch data: {e}")
            all_x, all_y, all_z = [], [], []

        # ===== STEP 5: ANALYZE =====
        if len(all_x) < 128:
            st.error(
                f"Not enough data! Got {len(all_x)} readings, need 128+\n\n"
                f"Try recording for longer or check connection"
            )
        else:
            status.success(
                f"Recorded {len(all_x)} readings! " f"(~{len(all_x)/duration:.0f} Hz)"
            )

            # Convert to numpy
            raw = np.column_stack([all_x, all_y, all_z])

            # Show raw data
            with st.expander("View Raw Phone Data"):
                fig = make_subplots(
                    rows=3,
                    cols=1,
                    shared_xaxes=True,
                    subplot_titles=("X-axis", "Y-axis", "Z-axis"),
                )
                # Only show first 1000 points for speed
                show_n = min(1000, len(all_x))
                fig.add_trace(
                    go.Scatter(y=all_x[:show_n], name="X", line=dict(color="red")),
                    row=1,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(y=all_y[:show_n], name="Y", line=dict(color="green")),
                    row=2,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(y=all_z[:show_n], name="Z", line=dict(color="blue")),
                    row=3,
                    col=1,
                )
                fig.update_layout(
                    height=500, title_text=f"Phone Sensor Data ({len(all_x)} readings)"
                )
                st.plotly_chart(fig, use_container_width=True)

            # ===== PREDICT =====
            with st.spinner("AI is analyzing your motion..."):
                result = predict(raw, encoder, classifier, label_map, input_hz=phone_hz)

            if result:
                emoji = EMOJI_MAP.get(result["activity"], "")

                # Big result display
                if result["is_fall"]:
                    st.error(
                        f"## FALL DETECTED!\n"
                        f"### Confidence: {result['confidence']:.1f}%"
                    )
                else:
                    st.success(
                        f"## {emoji} {result['activity']} Detected!\n"
                        f"### Confidence: {result['confidence']:.1f}%"
                    )

                # Metrics row
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Activity", result["activity"])
                col2.metric("Confidence", f"{result['confidence']:.1f}%")
                col3.metric("Data Points", len(all_x))
                col4.metric("Windows", result["num_windows"])

                # All predictions bar chart
                st.subheader("All Activity Probabilities")
                pred_df = pd.DataFrame(
                    {
                        "Activity": list(result["all_predictions"].keys()),
                        "Confidence": list(result["all_predictions"].values()),
                    }
                )

                fig = go.Figure(
                    data=[
                        go.Bar(
                            x=pred_df["Activity"],
                            y=pred_df["Confidence"],
                            marker_color=[
                                (
                                    "red"
                                    if a == "Fall"
                                    else (
                                        "green"
                                        if a == result["activity"]
                                        else "steelblue"
                                    )
                                )
                                for a in pred_df["Activity"]
                            ],
                            text=[f"{c:.1f}%" for c in pred_df["Confidence"]],
                            textposition="auto",
                        )
                    ]
                )
                fig.update_layout(
                    title="Activity Probabilities",
                    xaxis_title="Activity",
                    yaxis_title="Confidence (%)",
                    yaxis_range=[0, 100],
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)

                # Per window timeline
                with st.expander(
                    f"Per-Window Analysis ({result['num_windows']} windows)"
                ):
                    for i, wr in enumerate(result["per_window"]):
                        e = EMOJI_MAP.get(wr["activity"], "")
                        bar_len = int(wr["confidence"] / 5)
                        bar = "█" * bar_len
                        st.write(
                            f"Window {i+1}: {e} **{wr['activity']}** "
                            f"({wr['confidence']:.1f}%) {bar}"
                        )


# ========================
# Patients page
# ========================
def patients_page():
    st.markdown('<h1 class="main-header">Patients</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Manage and view all your patients</p>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        search = st.text_input(
            "Search patients...",
            placeholder="Search by name, ID, or condition",
            label_visibility="collapsed",
            key="patient_search",
        )
    with col2:
        filter_status = st.selectbox(
            "Status",
            ["All", "Active", "Inactive"],
            label_visibility="collapsed",
            key="status_filter",
        )
    with col3:
        if st.button(
            "+ Add Patient", width="stretch", type="primary", key="add_patient_btn"
        ):
            st.session_state.show_add_patient_form = True

    if st.session_state.show_add_patient_form:
        st.markdown("---")
        add_patient_form()
        st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["All Patients", "Active", "Inactive"])

    with tab1:
        display_patients(search, "all")
    with tab2:
        display_patients(search, "active")
    with tab3:
        display_patients(search, "inactive")


def display_patients(search, status_filter):
    patients = st.session_state.patients.copy()

    if search:
        patients = [
            p
            for p in patients
            if search.lower() in p["name"].lower()
            or search.lower() in p["id"].lower()
            or search.lower() in p["condition"].lower()
        ]

    if status_filter == "active":
        patients = [p for p in patients if p["status"] == "active"]
    elif status_filter == "inactive":
        patients = [p for p in patients if p["status"] == "inactive"]

    if not patients:
        return

    for patient in patients:
        unique_id = patient.get("unique_key", patient["id"].replace("#", ""))
        delete_key = f"del_{unique_id}_{status_filter}"
        view_key = f"view_{unique_id}_{status_filter}"

        status = patient["status"]
        if status == "active":
            status_html = '<span class="status-badge-active">Active</span>'
        else:
            status_html = '<span class="status-badge-inactive">Inactive</span>'

        col1, col2, col3, col4, col5 = st.columns([3, 3, 1.2, 0.7, 1.2])

        with col1:
            st.markdown(
                f'<p class="patient-name">{patient["name"]}</p>', unsafe_allow_html=True
            )
            st.markdown(
                f'<p class="patient-id">{patient["id"]}</p>', unsafe_allow_html=True
            )

        with col2:
            st.markdown(
                f'<p class="patient-condition">{patient["condition"]}</p>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<p class="patient-details">Age: {patient["age"]} | {patient["gender"]}</p>',
                unsafe_allow_html=True,
            )

        with col3:
            st.markdown(
                f'<p class="last-visit-label">Last Visit</p>', unsafe_allow_html=True
            )
            st.markdown(
                f'<p class="last-visit-value">{patient["last_visit"]}</p>',
                unsafe_allow_html=True,
            )
            st.markdown('<div class="small-delete-btn">', unsafe_allow_html=True)
            if st.button("Delete", key=delete_key):
                delete_patient(patient["id"])
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        with col4:
            st.markdown(f"{status_html}", unsafe_allow_html=True)

        with col5:
            st.markdown('<div class="small-details-btn">', unsafe_allow_html=True)
            if st.button("Details", key=view_key, width="stretch"):
                patient_details_modal(patient)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<hr class="patient-divider">', unsafe_allow_html=True)

    total = len(st.session_state.patients)
    showing = len(patients)
    st.markdown(
        f'<p style="font-size: 0.75rem; color: #6b7280; text-align: right; margin-top: 16px;">Showing {showing} of {total} patients</p>',
        unsafe_allow_html=True,
    )


# ========================
# FIXED ANALYZER PAGE with ML Integration
# ========================
def analyzer_page():
    st.markdown(
        '<h1 class="main-header">Activity Detection Analyzer</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="sub-header">Upload sensor data for activity detection</p>',
        unsafe_allow_html=True,
    )

    # Load models
    encoder, classifier, label_map = load_models()

    if encoder is None:
        st.error("⚠️ ML models not loaded! Check model paths.")
        return

    # File upload
    col1, col2 = st.columns(2)

    with col1:
        uploaded_file = st.file_uploader(
            "Upload CSV file with accelerometer data",
            type=["csv"],
            help="CSV with columns: accel_x, accel_y, accel_z (or any 3 numeric columns)",
            key="activity_csv_uploader",
        )

        input_hz = st.number_input(
            "Sampling Rate (Hz)",
            min_value=10,
            max_value=500,
            value=50,
            help="How many readings per second",
        )

    with col2:
        st.markdown("**Expected CSV Format:**")
        st.code(
            """accel_x,accel_y,accel_z
0.52,0.31,9.81
0.54,0.33,9.79
...(128+ rows needed)"""
        )

    if uploaded_file is not None:
        # Load data
        data = pd.read_csv(uploaded_file)

        st.success(f"✅ Loaded {len(data)} rows")

        # Find numeric columns
        if "accel_x" in data.columns:
            raw = data[["accel_x", "accel_y", "accel_z"]].values
        else:
            numeric = data.select_dtypes(include=[np.number])
            raw = numeric.iloc[:, :3].values
            st.info(f"Using columns: {list(numeric.columns[:3])}")

        # Show raw data
        with st.expander("View Raw Data"):
            st.dataframe(data.head(20))

            fig = make_subplots(
                rows=3,
                cols=1,
                shared_xaxes=True,
                subplot_titles=("X-axis", "Y-axis", "Z-axis"),
            )
            fig.add_trace(
                go.Scatter(y=raw[:500, 0], name="X", line=dict(color="red")),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(y=raw[:500, 1], name="Y", line=dict(color="green")),
                row=2,
                col=1,
            )
            fig.add_trace(
                go.Scatter(y=raw[:500, 2], name="Z", line=dict(color="blue")),
                row=3,
                col=1,
            )
            fig.update_layout(height=500, title_text="Raw Sensor Data")
            st.plotly_chart(fig, use_container_width=True)

        # CRITICAL: Predict button
        if st.button("🔍 Detect Activity", type="primary", key="detect_activity_btn"):
            with st.spinner("Analyzing with ML model..."):
                result = predict(raw, encoder, classifier, label_map, input_hz)
                st.session_state.prediction_result = result

        # Show results
        if st.session_state.prediction_result is not None:
            result = st.session_state.prediction_result

            if result is None:
                st.error("❌ Not enough data! Need at least 128 samples")
            else:
                emoji = EMOJI_MAP.get(result["activity"], "📊")

                # Big result display
                if result["is_fall"]:
                    st.error(
                        f"## 🚨 FALL DETECTED! Confidence: {result['confidence']:.1f}%"
                    )
                else:
                    st.success(
                        f"## {emoji} {result['activity']} Detected! Confidence: {result['confidence']:.1f}%"
                    )

                # Metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Activity", result["activity"])
                col2.metric("Confidence", f"{result['confidence']:.1f}%")
                col3.metric("Windows Analyzed", result["num_windows"])

                # All predictions bar chart
                st.subheader("All Predictions")
                pred_df = pd.DataFrame(
                    {
                        "Activity": list(result["all_predictions"].keys()),
                        "Confidence": list(result["all_predictions"].values()),
                    }
                )

                fig = go.Figure(
                    data=[
                        go.Bar(
                            x=pred_df["Activity"],
                            y=pred_df["Confidence"],
                            marker_color=[
                                "red" if a == "Fall" else "steelblue"
                                for a in pred_df["Activity"]
                            ],
                        )
                    ]
                )
                fig.update_layout(
                    title="Activity Probabilities",
                    xaxis_title="Activity",
                    yaxis_title="Confidence (%)",
                    yaxis_range=[0, 100],
                )
                st.plotly_chart(fig, use_container_width=True)

                # Per window results
                with st.expander(
                    f"Per-Window Analysis ({result['num_windows']} windows)"
                ):
                    for i, wr in enumerate(result["per_window"]):
                        e = EMOJI_MAP.get(wr["activity"], "📊")
                        st.write(
                            f"Window {i+1}: {e} {wr['activity']} ({wr['confidence']:.1f}%)"
                        )


# ========================
# Main app logic
# ========================
def main():
    if not st.session_state.logged_in:
        login_page()
    else:
        avatar_url = f"https://api.dicebear.com/7.x/initials/svg?seed={st.session_state.username}&backgroundColor=3b82f6"

        with st.sidebar:
            with st.expander(f"👤 {st.session_state.username}", expanded=False):
                st.markdown(
                    f"""
                <div style="display: flex; align-items: center; gap: 12px; padding: 8px 0;">
                    <img src="{avatar_url}" alt="Profile" style="border-radius: 50%; width: 48px; height: 48px; border: 2px solid #3b82f6;">
                    <div>
                        <p style="font-weight: 700; font-size: 1.05rem; color: white; margin: 0;">{st.session_state.username}</p>
                        <p style="font-size: 0.75rem; color: #9ca3af; margin: 2px 0 0 0;">Doctor</p>
                    </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                st.markdown("---")

                if st.button("🚪 Logout", use_container_width=True, key="logout_btn"):
                    st.session_state.logged_in = False
                    st.session_state.username = ""
                    st.rerun()

            st.markdown("---")

            pages = ["Dashboard", "Patients", "Analyzer"]
            for page_name in pages:
                if st.button(
                    page_name,
                    key=f"nav_{page_name}",
                    use_container_width=True,
                    type=(
                        "primary"
                        if st.session_state.current_page == page_name
                        else "secondary"
                    ),
                ):
                    st.session_state.current_page = page_name
                    st.rerun()

            st.markdown("---")

            st.markdown("### Quick Stats")
            st.markdown(f"Total Patients: **{len(st.session_state.patients)}**")
            active_count = len(
                [p for p in st.session_state.patients if p["status"] == "active"]
            )
            st.markdown(f"Active Patients: **{active_count}**")

        # Main content
        if st.session_state.current_page == "Dashboard":
            dashboard_page()
        elif st.session_state.current_page == "Patients":
            patients_page()
        elif st.session_state.current_page == "Analyzer":
            analyzer_page()


if __name__ == "__main__":
    main()
