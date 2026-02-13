"""
COMPLETE PIPELINE:
1. Load data with unified labels
2. Split into Train/Val/Test (per device)
3. Train multi-class classifier
4. Save training history (plots, metrics)
5. Create test CSV files for demo
6. Run demo using test files
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import pickle
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from scipy import signal
from sklearn.preprocessing import StandardScaler

from model import IMUEncoder


# ============================================
# CONFIGURATION
# ============================================

BASE_PATH = r"C:\Users\Amrit Shah\Desktop\protobyte 2.0\protobytes-2.0-team-Coder-Rex"
MODELS_PATH = os.path.join(BASE_PATH, "models")
DATA_PATH = os.path.join(BASE_PATH, "data", "processed")
DEMO_PATH = os.path.join(BASE_PATH, "demo_samples")
DOCS_PATH = os.path.join(BASE_PATH, "docs")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs(MODELS_PATH, exist_ok=True)
os.makedirs(DEMO_PATH, exist_ok=True)
os.makedirs(DOCS_PATH, exist_ok=True)


# Unified label mapping
UNIFIED_LABELS = {
    'Walking': 'Walking',
    'Running': 'Running',
    'Sitting': 'Sitting',
    'Standing': 'Standing',
    'Stairs_up': 'Stairs_Up',
    'Stairs_down': 'Stairs_Down',
    'Jumping': 'Jumping',
    'Lying': 'Lying',
    'Fall': 'Fall',
    'Other': 'Other',
    'Jogging': 'Running',
    'Upstairs': 'Stairs_Up',
    'Downstairs': 'Stairs_Down',
    'Cycling': 'Cycling',
    'Nordic_walking': 'Walking',
    'Vacuum_cleaning': 'Other',
    'Ironing': 'Other',
    'Rope_jumping': 'Jumping',
}


# ============================================
# DATASET CLASS
# ============================================

class ActivityDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# ============================================
# CLASSIFIER MODEL
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
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)


# ============================================
# STEP 1: LOAD AND UNIFY DATA
# ============================================

def load_and_unify():
    print("=" * 60)
    print("STEP 1: LOADING DATA")
    print("=" * 60)

    data_path = os.path.join(DATA_PATH, "combined_dataset.pkl")
    with open(data_path, 'rb') as f:
        dataset = pickle.load(f)

    X = dataset['data']
    original_labels = dataset['labels']
    sources = dataset['sources']

    # Unify labels
    unified = np.array([UNIFIED_LABELS.get(l, 'Other') for l in original_labels])

    # Show summary
    print("\nActivities:")
    for act in sorted(np.unique(unified)):
        count = np.sum(unified == act)
        srcs = np.unique(sources[unified == act])
        print(f"  {act:<15}: {count:>6} samples (from: {', '.join(srcs)})")

    print(f"\nTotal: {len(X)} samples")

    return X, unified, sources


# ============================================
# STEP 2: SPLIT DATA (Per Device)
# ============================================

def split_data(X, labels, sources):
    print("\n" + "=" * 60)
    print("STEP 2: SPLITTING DATA")
    print("=" * 60)

    train_X, train_y, train_src = [], [], []
    val_X, val_y, val_src = [], [], []
    test_X, test_y, test_src = [], [], []

    for source in ['SisFall', 'WISDM', 'PAMAP2']:
        mask = sources == source
        X_s = X[mask]
        y_s = labels[mask]
        src_s = sources[mask]

        # 70% train, 15% val, 15% test
        X_tr, X_temp, y_tr, y_temp, s_tr, s_temp = train_test_split(
            X_s, y_s, src_s, test_size=0.3, random_state=42, stratify=y_s
        )

        X_v, X_te, y_v, y_te, s_v, s_te = train_test_split(
            X_temp, y_temp, s_temp, test_size=0.5, random_state=42, stratify=y_temp
        )

        print(f"\n  {source}:")
        print(f"    Train: {len(X_tr):<6} | Val: {len(X_v):<6} | Test: {len(X_te)}")

        # Show per-activity split for this device
        for act in sorted(np.unique(y_s)):
            tr_count = np.sum(y_tr == act)
            v_count = np.sum(y_v == act)
            te_count = np.sum(y_te == act)
            if tr_count > 0:
                print(f"      {act:<15}: Train={tr_count:<5} Val={v_count:<5} Test={te_count}")

        train_X.append(X_tr); train_y.append(y_tr); train_src.append(s_tr)
        val_X.append(X_v); val_y.append(y_v); val_src.append(s_v)
        test_X.append(X_te); test_y.append(y_te); test_src.append(s_te)

    # Combine
    X_train = np.concatenate(train_X)
    y_train = np.concatenate(train_y)
    src_train = np.concatenate(train_src)
    X_val = np.concatenate(val_X)
    y_val = np.concatenate(val_y)
    src_val = np.concatenate(val_src)
    X_test = np.concatenate(test_X)
    y_test = np.concatenate(test_y)
    src_test = np.concatenate(test_src)

    print(f"\n  TOTAL:")
    print(f"    Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    return (X_train, y_train, src_train,
            X_val, y_val, src_val,
            X_test, y_test, src_test)


# ============================================
# STEP 3: TRAIN CLASSIFIER
# ============================================

def train(X_train, y_train, X_val, y_val):
    print("\n" + "=" * 60)
    print("STEP 3: TRAINING MULTI-CLASS CLASSIFIER")
    print("=" * 60)

    # Encode labels
    le = LabelEncoder()
    y_tr = le.fit_transform(y_train)
    y_v = le.transform(y_val)
    num_classes = len(le.classes_)

    # Save label map
    label_map = {str(i): str(c) for i, c in enumerate(le.classes_)}
    with open(os.path.join(MODELS_PATH, "label_map.json"), 'w') as f:
        json.dump(label_map, f, indent=2)

    print(f"\n  Classes ({num_classes}):")
    for i, c in enumerate(le.classes_):
        print(f"    {i}: {c} ({np.sum(y_train == c)} train samples)")

    # Dataloaders
    train_loader = DataLoader(
        ActivityDataset(X_train, y_tr), batch_size=128, shuffle=True
    )
    val_loader = DataLoader(
        ActivityDataset(X_val, y_v), batch_size=128, shuffle=False
    )

    # Load frozen encoder
    print("\n  Loading pre-trained encoder...")
    encoder = IMUEncoder(input_channels=3, embedding_dim=128)
    encoder.load_state_dict(torch.load(
        os.path.join(MODELS_PATH, "encoder.pth"), map_location=DEVICE
    ))
    encoder = encoder.to(DEVICE)
    encoder.eval()
    print("  Encoder loaded ✅ (frozen)")

    # Classifier
    classifier = ActivityClassifier(embedding_dim=128, num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    # Training history
    history = {
        'train_loss': [],
        'val_accuracy': [],
        'val_loss': [],
        'best_accuracy': 0,
        'best_epoch': 0
    }

    best_state = None

    print("\n  Training...")
    for epoch in range(50):
        # --- Train ---
        classifier.train()
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0

        for data, labels in train_loader:
            data = data.to(DEVICE)
            labels = labels.to(DEVICE)

            with torch.no_grad():
                emb = encoder.get_embedding(data)

            out = classifier(emb)
            loss = criterion(out, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            preds = torch.argmax(out, dim=1)
            epoch_correct += (preds == labels).sum().item()
            epoch_total += labels.size(0)

        train_loss = epoch_loss / len(train_loader)
        train_acc = 100 * epoch_correct / epoch_total

        # --- Validate ---
        classifier.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for data, labels in val_loader:
                data = data.to(DEVICE)
                labels = labels.to(DEVICE)

                emb = encoder.get_embedding(data)
                out = classifier(emb)
                loss = criterion(out, labels)

                val_loss += loss.item()
                preds = torch.argmax(out, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total

        scheduler.step(val_acc)

        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)

        if val_acc > history['best_accuracy']:
            history['best_accuracy'] = val_acc
            history['best_epoch'] = epoch + 1
            best_state = classifier.state_dict().copy()

        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1}/50 | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Best: {history['best_accuracy']:.2f}%")

    # Save best model
    classifier.load_state_dict(best_state)
    torch.save(classifier.state_dict(), os.path.join(MODELS_PATH, "multiclass_classifier.pth"))

    print(f"\n  ✅ Best Val Accuracy: {history['best_accuracy']:.2f}% (Epoch {history['best_epoch']})")

    # Save history
    with open(os.path.join(MODELS_PATH, "training_history.json"), 'w') as f:
        json.dump(history, f, indent=2)

    return encoder, classifier, le, history


# ============================================
# STEP 4: SAVE TRAINING HISTORY (PLOTS)
# ============================================

def save_training_plots(history):
    print("\n" + "=" * 60)
    print("STEP 4: SAVING TRAINING HISTORY")
    print("=" * 60)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Plot 1: Training Loss
    axes[0].plot(history['train_loss'], 'b-', linewidth=2, label='Train Loss')
    axes[0].plot(history['val_loss'], 'r-', linewidth=2, label='Val Loss')
    axes[0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend(fontsize=12)
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Validation Accuracy
    axes[1].plot(history['val_accuracy'], 'g-', linewidth=2)
    axes[1].axhline(y=history['best_accuracy'], color='orange', linestyle='--',
                    label=f"Best: {history['best_accuracy']:.2f}%")
    axes[1].set_title('Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend(fontsize=12)
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Summary text
    axes[2].axis('off')
    summary = f"""
    Training Summary
    ─────────────────────
    Total Epochs: {len(history['train_loss'])}
    Best Epoch: {history['best_epoch']}

    Train Loss:
      Start: {history['train_loss'][0]:.4f}
      End:   {history['train_loss'][-1]:.4f}

    Val Accuracy:
      Start: {history['val_accuracy'][0]:.2f}%
      Best:  {history['best_accuracy']:.2f}%
      End:   {history['val_accuracy'][-1]:.2f}%
    """
    axes[2].text(0.1, 0.5, summary, fontsize=14, fontfamily='monospace',
                 verticalalignment='center',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.suptitle('Multi-Class Activity Classifier — Training History',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    path = os.path.join(DOCS_PATH, "multiclass_training_history.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  ✅ Saved: {path}")


# ============================================
# STEP 5: EVALUATE ON TEST SET
# ============================================

def evaluate(encoder, classifier, le, X_test, y_test, src_test):
    print("\n" + "=" * 60)
    print("STEP 5: EVALUATION ON TEST SET")
    print("=" * 60)

    y_te = le.transform(y_test)
    test_loader = DataLoader(
        ActivityDataset(X_test, y_te), batch_size=128, shuffle=False
    )

    classifier.eval()
    encoder.eval()

    all_preds = []
    all_true = []

    idx = 0
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(DEVICE)
            emb = encoder.get_embedding(data)
            out = classifier(emb)
            preds = torch.argmax(out, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_true.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_true = np.array(all_true)

    # Overall accuracy
    overall = 100 * np.sum(all_preds == all_true) / len(all_true)
    print(f"\n  Overall Test Accuracy: {overall:.2f}%")

    # Per-device accuracy
    print(f"\n  Per-Device Accuracy:")
    idx = 0
    device_results = {}
    for source in ['SisFall', 'WISDM', 'PAMAP2']:
        mask = src_test == source
        count = mask.sum()
        src_preds = all_preds[idx:idx+count] if False else []  # We need different approach

    # Recalculate per device
    # We need to track indices properly
    test_loader2 = DataLoader(
        ActivityDataset(X_test, y_te), batch_size=128, shuffle=False
    )

    all_preds2 = []
    all_true2 = []
    batch_idx = 0

    with torch.no_grad():
        for data, labels in test_loader2:
            data = data.to(DEVICE)
            emb = encoder.get_embedding(data)
            out = classifier(emb)
            preds = torch.argmax(out, dim=1)
            all_preds2.extend(preds.cpu().numpy())
            all_true2.extend(labels.numpy())

    all_preds2 = np.array(all_preds2)
    all_true2 = np.array(all_true2)

    for source in ['SisFall', 'WISDM', 'PAMAP2']:
        mask = src_test == source
        if mask.sum() == 0:
            continue
        src_preds = all_preds2[mask]
        src_true = all_true2[mask]
        acc = 100 * np.sum(src_preds == src_true) / len(src_true)
        device_results[source] = acc
        print(f"    {source:<10}: {acc:.2f}% ({mask.sum()} samples)")

    # Per-class accuracy
    print(f"\n  Per-Activity Accuracy:")
    class_results = {}
    for i, cls in enumerate(le.classes_):
        mask = all_true2 == i
        if mask.sum() == 0:
            continue
        acc = 100 * np.sum(all_preds2[mask] == all_true2[mask]) / mask.sum()
        class_results[cls] = acc
        print(f"    {cls:<15}: {acc:.2f}% ({mask.sum()} samples)")

    # Classification report
    print(f"\n  Classification Report:")
    print(classification_report(all_true2, all_preds2, target_names=le.classes_))

    # Confusion matrix
    cm = confusion_matrix(all_true2, all_preds2)
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
    ax.set_title('Multi-Class Confusion Matrix', fontsize=16, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=14)
    ax.set_ylabel('Actual', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    path = os.path.join(DOCS_PATH, "multiclass_confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"\n  ✅ Confusion matrix saved: {path}")

    # Per-device accuracy bar chart
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    devices = list(device_results.keys())
    accs = list(device_results.values())
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars = ax.bar(devices, accs, color=colors, width=0.5, edgecolor='black')
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{acc:.1f}%', ha='center', fontsize=14, fontweight='bold')
    ax.set_title('Per-Device Accuracy', fontsize=16, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=14)
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    path = os.path.join(DOCS_PATH, "multiclass_per_device_accuracy.png")
    plt.savefig(path, dpi=150)
    print(f"  ✅ Per-device chart saved: {path}")

    # Save all results
    results = {
        'overall_accuracy': overall,
        'device_accuracy': device_results,
        'class_accuracy': class_results
    }
    with open(os.path.join(MODELS_PATH, "evaluation_results.json"), 'w') as f:
        json.dump(results, f, indent=2)

    return results


# ============================================
# STEP 6: CREATE TEST CSV FILES FOR DEMO
# ============================================

def create_test_files(X_test, y_test, src_test):
    print("\n" + "=" * 60)
    print("STEP 6: CREATING TEST FILES FOR DEMO")
    print("=" * 60)

    # For each activity + device combination
    for activity in sorted(np.unique(y_test)):
        for source in ['SisFall', 'WISDM', 'PAMAP2']:
            mask = (y_test == activity) & (src_test == source)
            indices = np.where(mask)[0]

            if len(indices) < 3:
                continue

            # Take 5 windows and concatenate
            num = min(5, len(indices))
            sample = np.concatenate([X_test[indices[i]] for i in range(num)], axis=0)

            filename = f"test_{activity.lower()}_{source.lower()}.csv"
            filepath = os.path.join(DEMO_PATH, filename)

            df = pd.DataFrame(sample, columns=['accel_x', 'accel_y', 'accel_z'])
            df.to_csv(filepath, index=False)

    # Count files
    files = sorted([f for f in os.listdir(DEMO_PATH) if f.endswith('.csv')])
    print(f"\n  Created {len(files)} test files:")

    for f in files:
        size = os.path.getsize(os.path.join(DEMO_PATH, f)) / 1024
        print(f"    📄 {f} ({size:.1f} KB)")

    print(f"\n  ✅ All saved in: {DEMO_PATH}")


# ============================================
# STEP 7: DEMO - PREDICT FROM CSV FILE
# ============================================

def predict_from_csv(filepath, encoder, classifier, label_map, input_hz=50):
    """
    Give a CSV file → Get activity prediction

    This is what you use during presentation!
    """

    # Load CSV
    data = pd.read_csv(filepath)

    if 'accel_x' in data.columns:
        raw = data[['accel_x', 'accel_y', 'accel_z']].values
    else:
        raw = data.select_dtypes(include=[np.number]).iloc[:, :3].values

    # Resample if needed
    if input_hz != 50:
        target_len = int(len(raw) * 50 / input_hz)
        resampled = np.zeros((target_len, 3))
        for col in range(3):
            resampled[:, col] = signal.resample(raw[:, col], target_len)
        raw = resampled

    # Create windows
    windows = []
    for start in range(0, len(raw) - 128, 64):
        window = raw[start:start + 128]
        scaler = StandardScaler()
        window = scaler.fit_transform(window)
        windows.append(window)

    if not windows:
        return {"error": "Not enough data!"}

    windows = np.array(windows, dtype=np.float32)

    # Predict
    with torch.no_grad():
        tensor = torch.tensor(windows, dtype=torch.float32).to(DEVICE)
        emb = encoder.get_embedding(tensor)
        out = classifier(emb)
        probs = torch.softmax(out, dim=1)
        preds = torch.argmax(out, dim=1)

    # Per-window results
    window_results = []
    for i in range(len(windows)):
        act = label_map[str(preds[i].item())]
        conf = probs[i][preds[i]].item() * 100
        window_results.append({'activity': act, 'confidence': conf})

    # Overall result (majority vote)
    activities = [r['activity'] for r in window_results]
    activity_counts = {}
    for a in activities:
        activity_counts[a] = activity_counts.get(a, 0) + 1

    dominant = max(activity_counts, key=activity_counts.get)

    # Average probabilities
    avg_probs = probs.mean(dim=0).cpu().numpy()
    all_predictions = {}
    for idx_str, name in label_map.items():
        all_predictions[name] = round(float(avg_probs[int(idx_str)]) * 100, 2)

    all_predictions = dict(sorted(all_predictions.items(), key=lambda x: -x[1]))

    return {
        'activity': dominant,
        'confidence': round(max(all_predictions.values()), 2),
        'all_predictions': all_predictions,
        'is_fall': dominant == 'Fall',
        'per_window': window_results,
        'activity_distribution': activity_counts,
        'num_windows': len(windows)
    }


# ============================================
# STEP 8: INTERACTIVE DEMO
# ============================================

def run_demo(encoder, classifier, label_map):
    print("\n" + "=" * 60)
    print("  🩺 ACTIVITY DETECTION DEMO")
    print("=" * 60)

    emoji_map = {
        'Walking': '🚶', 'Running': '🏃', 'Sitting': '🪑',
        'Standing': '🧍', 'Lying': '🛏️', 'Fall': '🚨',
        'Jumping': '🤸', 'Cycling': '🚴', 'Stairs_Up': '⬆️',
        'Stairs_Down': '⬇️', 'Other': '❓'
    }

    while True:
        print(f"\n  Options:")
        print(f"    1. Pick a test file to analyze")
        print(f"    2. Test ALL files (show accuracy)")
        print(f"    3. Cross-device comparison")
        print(f"    4. Enter custom CSV path")
        print(f"    5. Analyze Phyphox data")
        print(f"    q. Quit")

        choice = input("\n  Choice: ").strip()

        if choice == 'q':
            break

        elif choice == '1':
            # Show available files
            files = sorted([f for f in os.listdir(DEMO_PATH) if f.endswith('.csv')])

            print(f"\n  Available test files:")
            for i, f in enumerate(files):
                print(f"    {i+1}. {f}")

            file_num = input(f"\n  Enter file number (1-{len(files)}): ").strip()

            try:
                idx = int(file_num) - 1
                filepath = os.path.join(DEMO_PATH, files[idx])
            except:
                print("  ❌ Invalid number!")
                continue

            print(f"\n  Analyzing: {files[idx]}")

            result = predict_from_csv(filepath, encoder, classifier, label_map)

            if 'error' in result:
                print(f"  ❌ {result['error']}")
                continue

            # Display
            e = emoji_map.get(result['activity'], '📊')

            if result['is_fall']:
                print(f"""
    ╔══════════════════════════════════════════════╗
    ║  🚨🚨🚨 FALL DETECTED! 🚨🚨🚨              ║
    ║  Confidence: {result['confidence']:.1f}%                        ║
    ╚══════════════════════════════════════════════╝""")
            else:
                print(f"""
    ╔══════════════════════════════════════════════╗
    ║  {e} Detected: {result['activity']:<15}                ║
    ║  Confidence: {result['confidence']:.1f}%                        ║
    ║  Status: Normal ✅                           ║
    ╚══════════════════════════════════════════════╝""")

            print(f"\n  All Predictions:")
            for act, conf in result['all_predictions'].items():
                bar = '█' * int(conf / 3)
                e2 = emoji_map.get(act, '📊')
                print(f"    {e2} {act:<15} {bar} {conf:.1f}%")

            print(f"\n  Per-Window ({result['num_windows']} windows):")
            for i, wr in enumerate(result['per_window'][:10]):
                e2 = emoji_map.get(wr['activity'], '📊')
                print(f"    Window {i+1}: {e2} {wr['activity']:<15} {wr['confidence']:.1f}%")

            if len(result['per_window']) > 10:
                print(f"    ... and {len(result['per_window'])-10} more windows")

        elif choice == '2':
            files = sorted([f for f in os.listdir(DEMO_PATH) if f.endswith('.csv')])

            print(f"\n  Testing {len(files)} files...\n")
            print(f"  {'File':<45} {'Expected':<12} {'Detected':<12} {'Conf':<8} {'OK?'}")
            print(f"  {'─'*85}")

            correct = 0
            total = 0

            for filename in files:
                filepath = os.path.join(DEMO_PATH, filename)
                result = predict_from_csv(filepath, encoder, classifier, label_map)

                if 'error' in result:
                    continue

                # Get expected from filename
                parts = filename.replace('test_', '').replace('.csv', '').split('_')
                expected = parts[0].capitalize()

                detected = result['activity']
                conf = result['confidence']

                is_correct = (expected.lower() == detected.lower() or
                             expected.lower() in detected.lower() or
                             detected.lower() in expected.lower())

                mark = "✅" if is_correct else "❌"
                total += 1
                if is_correct:
                    correct += 1

                e = emoji_map.get(detected, '📊')
                print(f"  {filename:<45} {expected:<12} {e}{detected:<10} {conf:.1f}%   {mark}")

            if total > 0:
                print(f"\n  Demo Accuracy: {correct}/{total} = {100*correct/total:.1f}%")

        elif choice == '3':
            files = sorted([f for f in os.listdir(DEMO_PATH) if f.endswith('.csv')])

            # Group by activity
            groups = {}
            for f in files:
                parts = f.replace('test_', '').replace('.csv', '').split('_')
                act = parts[0].capitalize()
                if act not in groups:
                    groups[act] = []
                groups[act].append(f)

            print("\n  CROSS-DEVICE COMPARISON:")
            print("  Same activity → Different devices → Same detection?\n")

            for act, act_files in sorted(groups.items()):
                if len(act_files) < 2:
                    continue

                e = emoji_map.get(act, '📊')
                print(f"  {e} {act}:")

                for filename in act_files:
                    filepath = os.path.join(DEMO_PATH, filename)
                    result = predict_from_csv(filepath, encoder, classifier, label_map)

                    if 'error' in result:
                        continue

                    device = filename.replace(f'test_{act.lower()}_', '').replace('.csv', '')
                    detected = result['activity']
                    conf = result['confidence']
                    e2 = emoji_map.get(detected, '📊')
                    match = "✅" if detected == act else "❌"

                    print(f"    {device:<15} → {e2} {detected:<15} {conf:.1f}% {match}")

                print()

        elif choice == '4':
            filepath = input("  Enter CSV path: ").strip().strip('"')

            if not os.path.exists(filepath):
                print("  ❌ File not found!")
                continue

            result = predict_from_csv(filepath, encoder, classifier, label_map)

            if 'error' in result:
                print(f"  ❌ {result['error']}")
                continue

            e = emoji_map.get(result['activity'], '📊')
            print(f"\n  {e} Detected: {result['activity']} ({result['confidence']:.1f}%)")

            print(f"\n  All Predictions:")
            for act, conf in result['all_predictions'].items():
                e2 = emoji_map.get(act, '📊')
                print(f"    {e2} {act:<15}: {conf:.1f}%")

        elif choice == '5':
            print("\n  📱 PHYPHOX INSTRUCTIONS:")
            print("  1. Open Phyphox app")
            print("  2. Start 'Accelerometer' experiment")
            print("  3. Record activity for 5+ seconds")
            print("  4. Export → CSV")
            print("  5. Enter the file path below\n")

            filepath = input("  Phyphox CSV path: ").strip().strip('"')

            if not os.path.exists(filepath):
                print("  ❌ File not found!")
                continue

            # Phyphox default is ~100 Hz
            hz = input("  Sampling rate (default 100 Hz): ").strip()
            hz = int(hz) if hz else 100

            result = predict_from_csv(filepath, encoder, classifier, label_map, input_hz=hz)

            if 'error' in result:
                print(f"  ❌ {result['error']}")
                continue

            e = emoji_map.get(result['activity'], '📊')
            print(f"\n  {e} Detected: {result['activity']} ({result['confidence']:.1f}%)")


# ============================================
# MAIN
# ============================================

def main():
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║                                                        ║
    ║    🩺 Universal Activity Detection System               ║
    ║                                                        ║
    ║    1. Full Pipeline (Train + Evaluate + Demo)          ║
    ║    2. Demo Only (use existing model)                   ║
    ║                                                        ║
    ╚══════════════════════════════════════════════════════════╝
    """)

    choice = input("  Choice (1/2): ").strip()

    if choice == '1':
        # Step 1
        X, labels, sources = load_and_unify()

        # Step 2
        splits = split_data(X, labels, sources)
        X_train, y_train, src_train = splits[0], splits[1], splits[2]
        X_val, y_val, src_val = splits[3], splits[4], splits[5]
        X_test, y_test, src_test = splits[6], splits[7], splits[8]

        # Step 3
        encoder, classifier, le, history = train(X_train, y_train, X_val, y_val)

        # Step 4
        save_training_plots(history)

        # Step 5
        evaluate(encoder, classifier, le, X_test, y_test, src_test)

        # Step 6
        create_test_files(X_test, y_test, src_test)

        # Step 7
        label_map_path = os.path.join(MODELS_PATH, "label_map.json")
        with open(label_map_path, 'r') as f:
            label_map = json.load(f)

        num_classes = len(label_map)
        classifier2 = ActivityClassifier(embedding_dim=128, num_classes=num_classes)
        classifier2.load_state_dict(torch.load(
            os.path.join(MODELS_PATH, "multiclass_classifier.pth"), map_location=DEVICE
        ))
        classifier2 = classifier2.to(DEVICE)
        classifier2.eval()

        print("\n✅ EVERYTHING READY! Starting demo...\n")
        run_demo(encoder, classifier2, label_map)

    elif choice == '2':
        # Load existing models
        print("\nLoading models...")

        encoder = IMUEncoder(input_channels=3, embedding_dim=128)
        encoder.load_state_dict(torch.load(
            os.path.join(MODELS_PATH, "encoder.pth"), map_location=DEVICE
        ))
        encoder = encoder.to(DEVICE)
        encoder.eval()

        label_map_path = os.path.join(MODELS_PATH, "label_map.json")
        with open(label_map_path, 'r') as f:
            label_map = json.load(f)

        num_classes = len(label_map)
        classifier = ActivityClassifier(embedding_dim=128, num_classes=num_classes)
        classifier.load_state_dict(torch.load(
            os.path.join(MODELS_PATH, "multiclass_classifier.pth"), map_location=DEVICE
        ))
        classifier = classifier.to(DEVICE)
        classifier.eval()

        print("✅ Models loaded!")
        run_demo(encoder, classifier, label_map)


if __name__ == "__main__":
    main()