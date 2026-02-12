"""
Training script for the complete pipeline:
1. Train encoder with contrastive learning (self-supervised)
2. Train fall detection classifier (supervised)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pickle
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from model import IMUEncoder, ContrastiveLoss, FallDetectionClassifier
from dataset import ContrastiveIMUDataset, IMUDataset


def train_contrastive(model, dataloader, criterion, optimizer, device, epochs=50):
    """
    Phase 1: Train encoder using contrastive learning

    The model learns to create similar embeddings for similar motions
    WITHOUT using any labels (self-supervised)
    """

    model.train()
    all_losses = []

    print("\n" + "=" * 60)
    print("PHASE 1: CONTRASTIVE LEARNING (Self-Supervised)")
    print("No labels used! The model teaches itself.")
    print("=" * 60)

    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0

        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for view1, view2, labels, sources in progress:
            view1 = view1.to(device)
            view2 = view2.to(device)

            # Forward: get embeddings for both views
            z1 = model(view1)
            z2 = model(view2)

            # Compute contrastive loss
            loss = criterion(z1, z2)

            # Backward: update model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            progress.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = epoch_loss / num_batches
        all_losses.append(avg_loss)

        # Print every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs} - Avg Loss: {avg_loss:.4f}")

    return all_losses


def train_fall_classifier(
    encoder,
    classifier,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs=30,
):
    """
    Phase 2: Train fall detection classifier

    The encoder is FROZEN (not updated)
    Only the classifier learns to detect falls from embeddings
    """

    encoder.eval()  # Freeze encoder

    all_train_losses = []
    all_val_accuracies = []
    best_accuracy = 0

    print("\n" + "=" * 60)
    print("PHASE 2: FALL DETECTION (Supervised)")
    print("Encoder is frozen. Only classifier is learning.")
    print("=" * 60)

    label_map = {"Fall": 1, "Non-Fall": 0}

    for epoch in range(epochs):
        # --- Training ---
        classifier.train()
        epoch_loss = 0.0
        num_batches = 0

        for data, labels, sources in train_loader:
            data = data.to(device)
            label_tensor = torch.tensor([label_map.get(l, 0) for l in labels]).to(
                device
            )

            # Get embeddings from frozen encoder
            with torch.no_grad():
                embeddings = encoder.get_embedding(data)

            # Classify
            outputs = classifier(embeddings)
            loss = criterion(outputs, label_tensor)

            # Update classifier only
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        all_train_losses.append(avg_loss)

        # --- Validation ---
        classifier.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, labels, sources in val_loader:
                data = data.to(device)
                label_tensor = torch.tensor([label_map.get(l, 0) for l in labels]).to(
                    device
                )

                embeddings = encoder.get_embedding(data)
                outputs = classifier(embeddings)
                predictions = torch.argmax(outputs, dim=1)

                total += label_tensor.size(0)
                correct += (predictions == label_tensor).sum().item()

        accuracy = 100 * correct / total
        all_val_accuracies.append(accuracy)

        if accuracy > best_accuracy:
            best_accuracy = accuracy

        # Print every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(
                f"  Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Val Acc: {accuracy:.2f}% - Best: {best_accuracy:.2f}%"
            )

    return all_train_losses, all_val_accuracies


def main():
    """Main training pipeline"""

    start_time = time.time()

    # ============================================
    # SETUP
    # ============================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths
    base_path = (
        r"C:\Users\Amrit Shah\Desktop\protobyte 2.0\protobytes-2.0-team-Coder-Rex"
    )
    data_path = os.path.join(base_path, "data", "processed", "combined_dataset.pkl")
    models_path = os.path.join(base_path, "models")
    docs_path = os.path.join(base_path, "docs")

    os.makedirs(models_path, exist_ok=True)
    os.makedirs(docs_path, exist_ok=True)

    # ============================================
    # LOAD DATA
    # ============================================

    print("\nLoading processed data...")
    with open(data_path, "rb") as f:
        dataset = pickle.load(f)

    X = dataset["data"]
    y = dataset["binary_labels"]
    sources = dataset["sources"]

    print(f"Total samples: {len(X)}")
    print(f"Shape: {X.shape}")
    print(f"Falls: {np.sum(y == 'Fall')}")
    print(f"Non-Falls: {np.sum(y == 'Non-Fall')}")

    # ============================================
    # SPLIT DATA
    # ============================================

    print("\nSplitting data...")
    X_train, X_test, y_train, y_test, src_train, src_test = train_test_split(
        X, y, sources, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train: {len(X_train)} samples")
    print(f"Test:  {len(X_test)} samples")

    # ============================================
    # CREATE DATALOADERS
    # ============================================

    print("\nCreating dataloaders...")

    # For contrastive learning (with augmentation)
    contrastive_dataset = ContrastiveIMUDataset(X_train, y_train, src_train)
    contrastive_loader = DataLoader(
        contrastive_dataset, batch_size=128, shuffle=True, num_workers=0
    )

    # For classifier training (no augmentation)
    train_dataset = IMUDataset(X_train, y_train, src_train)
    train_loader = DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=0
    )

    # For validation
    test_dataset = IMUDataset(X_test, y_test, src_test)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

    print(f"Contrastive batches: {len(contrastive_loader)}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # ============================================
    # PHASE 1: CONTRASTIVE LEARNING
    # ============================================

    encoder = IMUEncoder(input_channels=3, embedding_dim=128).to(device)
    contrastive_criterion = ContrastiveLoss(temperature=0.5)
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=0.001)

    contrastive_losses = train_contrastive(
        model=encoder,
        dataloader=contrastive_loader,
        criterion=contrastive_criterion,
        optimizer=encoder_optimizer,
        device=device,
        epochs=30,
    )

    # Save encoder
    encoder_path = os.path.join(models_path, "encoder.pth")
    torch.save(encoder.state_dict(), encoder_path)
    print(f"\n✅ Encoder saved to: {encoder_path}")

    # ============================================
    # PHASE 2: FALL DETECTION CLASSIFIER
    # ============================================

    classifier = FallDetectionClassifier(embedding_dim=128, num_classes=2).to(device)
    classifier_criterion = nn.CrossEntropyLoss()
    classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

    train_losses, val_accuracies = train_fall_classifier(
        encoder=encoder,
        classifier=classifier,
        train_loader=train_loader,
        val_loader=test_loader,
        criterion=classifier_criterion,
        optimizer=classifier_optimizer,
        device=device,
        epochs=30,
    )

    # Save classifier
    classifier_path = os.path.join(models_path, "classifier.pth")
    torch.save(classifier.state_dict(), classifier_path)
    print(f"\n✅ Classifier saved to: {classifier_path}")

    # ============================================
    # SAVE TRAINING PLOTS
    # ============================================

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Contrastive Loss
    axes[0].plot(contrastive_losses, "b-", linewidth=2)
    axes[0].set_title("Contrastive Learning Loss", fontsize=14)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Classifier Loss
    axes[1].plot(train_losses, "r-", linewidth=2)
    axes[1].set_title("Classifier Training Loss", fontsize=14)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Validation Accuracy
    axes[2].plot(val_accuracies, "g-", linewidth=2)
    axes[2].set_title("Fall Detection Accuracy", fontsize=14)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Accuracy (%)")
    axes[2].grid(True, alpha=0.3)

    plt.suptitle("Training Results", fontsize=16, fontweight="bold")
    plt.tight_layout()

    plot_path = os.path.join(docs_path, "training_results.png")
    plt.savefig(plot_path, dpi=150)
    plt.show()
    print(f"\n✅ Training plots saved to: {plot_path}")

    # ============================================
    # FINAL SUMMARY
    # ============================================

    total_time = time.time() - start_time

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(
        f"""
    Total training time: {total_time/60:.1f} minutes
    
    Phase 1 (Contrastive):
      Initial loss: {contrastive_losses[0]:.4f}
      Final loss:   {contrastive_losses[-1]:.4f}
    
    Phase 2 (Fall Detection):
      Initial accuracy: {val_accuracies[0]:.2f}%
      Final accuracy:   {val_accuracies[-1]:.2f}%
      Best accuracy:    {max(val_accuracies):.2f}%
    
    Saved Models:
      Encoder:    {encoder_path}
      Classifier: {classifier_path}
    
    Saved Plots:
      {plot_path}
    """
    )


if __name__ == "__main__":
    main()
