"""
Preprocessing pipeline for SisFall, WISDM, and PAMAP2 datasets
Makes all 3 datasets compatible for contrastive learning
"""

import numpy as np
import pandas as pd
import os
from scipy import signal
from sklearn.preprocessing import StandardScaler


# ============================================
# STEP 1: LOAD EACH DATASET
# ============================================


def load_sisfall(data_path):
    """
    Load SisFall dataset

    SisFall has:
    - Folders: SA01-SA23 (young adults), SE01-SE15 (elderly)
    - Files: D01=activity, F01=fall
    - Columns: accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z
    - Sampling rate: 200 Hz
    """

    all_data = []
    all_labels = []

    # Activity mapping from filename
    # D = Daily activity, F = Fall
    activity_map = {
        "D01": "Walking",
        "D02": "Walking",
        "D03": "Walking",
        "D04": "Walking",
        "D05": "Standing",
        "D06": "Standing",
        "D07": "Standing",
        "D08": "Standing",
        "D09": "Sitting",
        "D10": "Sitting",
        "D11": "Stairs_up",
        "D12": "Stairs_down",
        "D13": "Running",
        "D14": "Running",
        "D15": "Jumping",
        "D16": "Jumping",
        "D17": "Other",
        "D18": "Other",
        "D19": "Other",
        "F01": "Fall",
        "F02": "Fall",
        "F03": "Fall",
        "F04": "Fall",
        "F05": "Fall",
        "F06": "Fall",
        "F07": "Fall",
        "F08": "Fall",
        "F09": "Fall",
        "F10": "Fall",
        "F11": "Fall",
        "F12": "Fall",
        "F13": "Fall",
        "F14": "Fall",
        "F15": "Fall",
    }

    subjects = sorted(
        [
            d
            for d in os.listdir(data_path)
            if os.path.isdir(os.path.join(data_path, d))
            and (d.startswith("SA") or d.startswith("SE"))
        ]
    )

    print(f"  Found {len(subjects)} subjects")

    for subject in subjects:
        subject_path = os.path.join(data_path, subject)
        files = [
            f
            for f in os.listdir(subject_path)
            if f.endswith(".txt") or f.endswith(".csv")
        ]

        for file in files:
            # Get activity code from filename (e.g., D01, F01)
            activity_code = file[:3]

            if activity_code not in activity_map:
                continue

            filepath = os.path.join(subject_path, file)

            try:
                # Try different separators
                try:
                    data = pd.read_csv(filepath, header=None, sep=",")
                except:
                    data = pd.read_csv(filepath, header=None, sep=";")

                # Take only first 3 columns (accelerometer x, y, z)
                if data.shape[1] >= 3:
                    accel_data = data.iloc[:, :3].values

                    # Remove any non-numeric values
                    accel_data = (
                        pd.DataFrame(accel_data)
                        .apply(pd.to_numeric, errors="coerce")
                        .dropna()
                        .values
                    )

                    if len(accel_data) > 0:
                        all_data.append(accel_data)
                        all_labels.append(activity_map[activity_code])

            except Exception as e:
                continue

    print(f"  Loaded {len(all_data)} recordings")
    return all_data, all_labels, 200  # 200 Hz sampling rate


def load_wisdm(data_path):
    """
    Load WISDM dataset

    WISDM has:
    - One big text file with all data
    - Columns: user, activity, timestamp, x, y, z
    - Sampling rate: 20 Hz
    """

    all_data = []
    all_labels = []

    # Find the data file
    data_file = None
    for f in os.listdir(data_path):
        if f.endswith(".txt") and "raw" in f.lower():
            data_file = os.path.join(data_path, f)
            break

    if data_file is None:
        # Try any txt file
        for f in os.listdir(data_path):
            if f.endswith(".txt"):
                data_file = os.path.join(data_path, f)
                break

    if data_file is None:
        print("  ERROR: No data file found in WISDM folder!")
        return all_data, all_labels, 20

    print(f"  Loading: {data_file}")

    # Read line by line (WISDM has messy formatting)
    rows = []
    with open(data_file, "r") as f:
        for line in f:
            line = line.strip().rstrip(";").rstrip(",")
            if not line:
                continue
            parts = line.split(",")
            if len(parts) >= 6:
                try:
                    user = int(parts[0])
                    activity = parts[1].strip()
                    x = float(parts[3])
                    y = float(parts[4])
                    z = float(parts[5].rstrip(";"))
                    rows.append([user, activity, x, y, z])
                except:
                    continue

    df = pd.DataFrame(rows, columns=["user", "activity", "x", "y", "z"])

    print(f"  Total rows: {len(df)}")
    print(f"  Activities: {df['activity'].unique()}")
    print(f"  Users: {df['user'].nunique()}")

    # Group by user and activity to create separate recordings
    for (user, activity), group in df.groupby(["user", "activity"]):
        accel_data = group[["x", "y", "z"]].values

        if len(accel_data) > 100:  # At least 100 samples
            all_data.append(accel_data)
            all_labels.append(activity)

    print(f"  Loaded {len(all_data)} recordings")
    return all_data, all_labels, 20  # 20 Hz sampling rate


def load_pamap2(data_path):
    """
    Load PAMAP2 dataset

    PAMAP2 has:
    - Protocol folder with subject101.dat to subject109.dat
    - 54 columns per file
    - Column 2: Activity ID
    - Columns 5,6,7: Hand accelerometer (x, y, z) ← WE USE THESE
    - Sampling rate: 100 Hz
    """

    all_data = []
    all_labels = []

    activity_map = {
        1: "Lying",
        2: "Sitting",
        3: "Standing",
        4: "Walking",
        5: "Running",
        6: "Cycling",
        7: "Walking",
        12: "Stairs_up",
        13: "Stairs_down",
        16: "Other",
        17: "Other",
        24: "Jumping",
    }

    protocol_path = os.path.join(data_path, "Protocol")

    if not os.path.exists(protocol_path):
        print(f"  ERROR: Protocol folder not found at {protocol_path}")
        return all_data, all_labels, 100

    files = sorted([f for f in os.listdir(protocol_path) if f.endswith(".dat")])
    print(f"  Found {len(files)} subject files")

    for file in files:
        filepath = os.path.join(protocol_path, file)

        try:
            data = pd.read_csv(filepath, sep=" ", header=None)

            # Column 1 = activity ID
            # Columns 4,5,6 = hand accelerometer x,y,z (0-indexed)

            for activity_id in data[1].unique():
                if activity_id not in activity_map:
                    continue

                # Filter data for this activity
                activity_data = data[data[1] == activity_id]

                # Extract hand accelerometer (columns 4, 5, 6)
                accel_data = activity_data[[4, 5, 6]].values

                # Remove NaN rows
                mask = ~np.isnan(accel_data).any(axis=1)
                accel_data = accel_data[mask]

                if len(accel_data) > 100:
                    all_data.append(accel_data)
                    all_labels.append(activity_map[activity_id])

        except Exception as e:
            print(f"  Error loading {file}: {e}")
            continue

    print(f"  Loaded {len(all_data)} recordings")
    return all_data, all_labels, 100  # 100 Hz sampling rate


# ============================================
# STEP 2: RESAMPLE TO SAME FREQUENCY
# ============================================


def resample_recording(data, original_hz, target_hz=50):
    """
    Resample a single recording to target frequency

    Example:
    SisFall 200 Hz → 50 Hz (take every 4th sample)
    WISDM 20 Hz → 50 Hz (interpolate to add more samples)
    PAMAP2 100 Hz → 50 Hz (take every 2nd sample)
    """

    if original_hz == target_hz:
        return data

    original_length = len(data)
    target_length = int(original_length * target_hz / original_hz)

    if target_length < 10:
        return None

    resampled = np.zeros((target_length, data.shape[1]))

    for col in range(data.shape[1]):
        resampled[:, col] = signal.resample(data[:, col], target_length)

    return resampled


# ============================================
# STEP 3: NORMALIZE VALUES
# ============================================


def normalize_recording(data):
    """
    Normalize to zero mean and unit variance

    Before: values could be 0.5 or 500 depending on device
    After: all values centered around 0 with similar range
    """

    scaler = StandardScaler()
    normalized = scaler.fit_transform(data)

    return normalized


# ============================================
# STEP 4: CUT INTO WINDOWS
# ============================================


def create_windows(data, window_size=128, step=64):
    """
    Cut a long recording into fixed-size windows

    Example:
    Recording: [1,2,3,4,5,6,7,8,9,10,11,12...]
    Window size: 4, Step: 2

    Window 1: [1,2,3,4]
    Window 2: [3,4,5,6]    (overlapping by 2)
    Window 3: [5,6,7,8]
    Window 4: [7,8,9,10]

    Why overlap? More training data + smoother transitions
    """

    windows = []

    for start in range(0, len(data) - window_size, step):
        window = data[start : start + window_size]
        windows.append(window)

    return windows


# ============================================
# STEP 5: COMPLETE PIPELINE
# ============================================


def preprocess_dataset(
    raw_data_list, labels_list, original_hz, target_hz=50, window_size=128, step=64
):
    """
    Complete preprocessing for one dataset

    Input: List of raw recordings + labels
    Output: Array of windows + labels
    """

    all_windows = []
    all_window_labels = []

    for recording, label in zip(raw_data_list, labels_list):

        # Step 2: Resample
        resampled = resample_recording(recording, original_hz, target_hz)

        if resampled is None or len(resampled) < window_size:
            continue

        # Step 3: Normalize
        normalized = normalize_recording(resampled)

        # Step 4: Create windows
        windows = create_windows(normalized, window_size, step)

        for window in windows:
            all_windows.append(window)
            all_window_labels.append(label)

    return np.array(all_windows), np.array(all_window_labels)


# ============================================
# MAIN: RUN EVERYTHING
# ============================================


def run_preprocessing():
    """Run complete preprocessing pipeline"""

    import pickle

    print("=" * 60)
    print("PREPROCESSING ALL DATASETS")
    print("=" * 60)

    base_path = (
        r"/home/ashutosh/Desktop/hackathon training/protobytes-2.0-team-Coder-Rex"
    )

    sisfall_path = os.path.join(base_path, "data", "raw", "SisFall")
    wisdm_path = os.path.join(base_path, "data", "raw", "WISDM")
    pamap2_path = os.path.join(base_path, "data", "raw", "PAMAP2")
    output_path = os.path.join(base_path, "data", "processed")

    os.makedirs(output_path, exist_ok=True)

    # ---- Load SisFall ----
    print("\n[1/6] Loading SisFall...")
    sisfall_data, sisfall_labels, sisfall_hz = load_sisfall(sisfall_path)

    # ---- Load WISDM ----
    print("\n[2/6] Loading WISDM...")
    wisdm_data, wisdm_labels, wisdm_hz = load_wisdm(wisdm_path)

    # ---- Load PAMAP2 ----
    print("\n[3/6] Loading PAMAP2...")
    pamap2_data, pamap2_labels, pamap2_hz = load_pamap2(pamap2_path)

    # ---- Preprocess SisFall ----
    print("\n[4/6] Preprocessing SisFall (200 Hz → 50 Hz)...")
    sisfall_windows, sisfall_window_labels = preprocess_dataset(
        sisfall_data, sisfall_labels, sisfall_hz
    )
    print(f"  Result: {sisfall_windows.shape}")

    # ---- Preprocess WISDM ----
    print("\n[5/6] Preprocessing WISDM (20 Hz → 50 Hz)...")
    wisdm_windows, wisdm_window_labels = preprocess_dataset(
        wisdm_data, wisdm_labels, wisdm_hz
    )
    print(f"  Result: {wisdm_windows.shape}")

    # ---- Preprocess PAMAP2 ----
    print("\n[6/6] Preprocessing PAMAP2 (100 Hz → 50 Hz)...")
    pamap2_windows, pamap2_window_labels = preprocess_dataset(
        pamap2_data, pamap2_labels, pamap2_hz
    )
    print(f"  Result: {pamap2_windows.shape}")

    # ---- Create source labels ----
    sisfall_sources = np.array(["SisFall"] * len(sisfall_windows))
    wisdm_sources = np.array(["WISDM"] * len(wisdm_windows))
    pamap2_sources = np.array(["PAMAP2"] * len(pamap2_windows))

    # ---- Combine all ----
    print("\n" + "=" * 60)
    print("COMBINING ALL DATASETS")
    print("=" * 60)

    all_windows = np.concatenate([sisfall_windows, wisdm_windows, pamap2_windows])
    all_labels = np.concatenate(
        [sisfall_window_labels, wisdm_window_labels, pamap2_window_labels]
    )
    all_sources = np.concatenate([sisfall_sources, wisdm_sources, pamap2_sources])

    # Create binary labels (Fall vs Non-Fall)
    binary_labels = np.array(
        ["Fall" if "Fall" in l else "Non-Fall" for l in all_labels]
    )

    print(f"\nFinal Dataset:")
    print(f"  Total windows: {len(all_windows)}")
    print(f"  Window shape: {all_windows[0].shape}")
    print(f"  Activities: {np.unique(all_labels, return_counts=True)}")
    print(f"  Sources: {np.unique(all_sources, return_counts=True)}")
    print(f"  Falls: {np.sum(binary_labels == 'Fall')}")
    print(f"  Non-Falls: {np.sum(binary_labels == 'Non-Fall')}")

    # ---- Save ----
    output_path = "../data/processed/"
    os.makedirs(output_path, exist_ok=True)

    processed_data = {
        "data": all_windows,
        "labels": all_labels,
        "binary_labels": binary_labels,
        "sources": all_sources,
    }

    save_path = os.path.join(output_path, "combined_dataset.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(processed_data, f)

    print(f"\n✅ Saved to: {save_path}")
    print(f"✅ File size: {os.path.getsize(save_path) / (1024*1024):.1f} MB")

    # ---- Save individual datasets too ----
    individual = {
        "sisfall": {"data": sisfall_windows, "labels": sisfall_window_labels},
        "wisdm": {"data": wisdm_windows, "labels": wisdm_window_labels},
        "pamap2": {"data": pamap2_windows, "labels": pamap2_window_labels},
    }

    individual_path = os.path.join(output_path, "individual_datasets.pkl")
    with open(individual_path, "wb") as f:
        pickle.dump(individual, f)

    print(f"✅ Individual datasets saved to: {individual_path}")

    return processed_data


if __name__ == "__main__":
    run_preprocessing()
