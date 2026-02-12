"""
Contrastive Learning Model for IMU data

Components:
1. IMUEncoder - Converts motion data → embedding
2. ContrastiveLoss - Teaches model to group similar motions
3. FallDetectionClassifier - Detects falls from embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class IMUEncoder(nn.Module):
    """
    1D CNN Encoder for IMU time series data

    Think of it like a funnel:

    Input:  (batch, 128 timesteps, 3 channels)  ← Wide
               ↓ Conv Layer 1
            (batch, 64 timesteps, 32 filters)
               ↓ Conv Layer 2
            (batch, 32 timesteps, 64 filters)
               ↓ Conv Layer 3
            (batch, 32 timesteps, 128 filters)
               ↓ Global Pool
            (batch, 128)                         ← Narrow
               ↓ Projection Head
    Output: (batch, 128)                         ← Embedding
    """

    def __init__(self, input_channels=3, embedding_dim=128):
        super(IMUEncoder, self).__init__()

        # Layer 1: Detect basic patterns (edges, peaks)
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=8, stride=1, padding=4)
        self.bn1 = nn.BatchNorm1d(32)

        # Layer 2: Detect medium patterns (steps, movements)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(64)

        # Layer 3: Detect complex patterns (walking, falling)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)

        # Squeeze everything into one value per filter
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Projection head (used ONLY during contrastive training)
        self.projection = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, embedding_dim)
        )

    def forward(self, x):
        """
        Full forward pass (used during contrastive training)
        Returns projected embedding
        """
        # Input x shape: (batch, 128 timesteps, 3 channels)
        # Conv1d needs: (batch, channels, timesteps)
        x = x.permute(0, 2, 1)

        # Conv Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, 2)  # 128 → 64

        # Conv Block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, 2)  # 64 → 32

        # Conv Block 3
        x = F.relu(self.bn3(self.conv3(x)))

        # Global pooling
        x = self.global_pool(x)  # (batch, 128, 1)
        x = x.squeeze(-1)  # (batch, 128)

        # Project
        embedding = self.projection(x)

        # Normalize (makes contrastive loss work better)
        embedding = F.normalize(embedding, dim=1)

        return embedding

    def get_embedding(self, x):
        """
        Get embedding WITHOUT projection head
        Used AFTER training for downstream tasks (fall detection, etc.)

        Why? The projection head is only useful during training.
        The real "knowledge" is in the conv layers.
        """
        x = x.permute(0, 2, 1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, 2)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, 2)

        x = F.relu(self.bn3(self.conv3(x)))

        x = self.global_pool(x)
        x = x.squeeze(-1)

        return x  # (batch, 128) - no projection


class ContrastiveLoss(nn.Module):
    """
    NT-Xent Loss (Normalized Temperature-scaled Cross Entropy)

    How it works (simple explanation):

    Batch of 4 samples: [A, B, C, D]
    Create pairs:       [A1, B1, C1, D1] and [A2, B2, C2, D2]

    The loss says:
    - A1 and A2 should be SIMILAR    (same sample, different augmentation)
    - A1 and B2 should be DIFFERENT  (different samples)
    - A1 and C2 should be DIFFERENT
    - A1 and D2 should be DIFFERENT

    Temperature controls how "strict" the model is:
    - Low temp (0.1) = very strict, hard to satisfy
    - High temp (1.0) = relaxed, easier
    - We use 0.5 = balanced
    """

    def __init__(self, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        """
        z1: embeddings of augmented view 1 (batch_size, embedding_dim)
        z2: embeddings of augmented view 2 (batch_size, embedding_dim)
        """
        batch_size = z1.shape[0]

        # Stack both views: [A1,B1,C1,D1,A2,B2,C2,D2]
        z = torch.cat([z1, z2], dim=0)

        # Compute similarity between ALL pairs
        # sim[i][j] = how similar sample i is to sample j
        sim = torch.mm(z, z.t()) / self.temperature

        # Mask out self-similarity (don't compare with yourself)
        mask = torch.eye(2 * batch_size, device=z.device).bool()
        sim = sim.masked_fill(mask, float("-inf"))

        # For each sample, its positive pair is batch_size positions away
        # A1's positive = A2 (at position 0+batch_size=4)
        # B1's positive = B2 (at position 1+batch_size=5)
        labels = torch.cat(
            [torch.arange(batch_size, 2 * batch_size), torch.arange(batch_size)], dim=0
        ).to(z.device)

        # Cross entropy: maximize similarity with positive pair
        # minimize similarity with all negative pairs
        loss = F.cross_entropy(sim, labels)

        return loss


class FallDetectionClassifier(nn.Module):
    """
    Simple classifier that sits ON TOP of the encoder

    Flow:
    Motion data → [Encoder (frozen)] → Embedding → [This Classifier] → Fall/No Fall

    The encoder is NOT trained here. Only this classifier is trained.
    This proves that the embeddings contain useful health information.
    """

    def __init__(self, embedding_dim=128, num_classes=2):
        super(FallDetectionClassifier, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),  # Prevents overfitting
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes),
        )

    def forward(self, embedding):
        return self.classifier(embedding)


# ============================================
# TEST (Run this file to verify it works)
# ============================================

if __name__ == "__main__":
    print("Testing model components...\n")

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create fake batch of IMU data
    # 32 samples, each with 128 timesteps and 3 channels
    batch_size = 32
    fake_data = torch.randn(batch_size, 128, 3).to(device)
    print(f"Input shape: {fake_data.shape}")

    # --- Test Encoder ---
    print("\n--- Testing IMU Encoder ---")
    encoder = IMUEncoder(input_channels=3, embedding_dim=128).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"Total parameters: {total_params:,}")

    # Forward pass
    embedding = encoder(fake_data)
    print(f"Projected embedding shape: {embedding.shape}")
    print(f"Embedding norm: {torch.norm(embedding[0]):.4f} (should be ~1.0)")

    # Get embedding without projection
    raw_embedding = encoder.get_embedding(fake_data)
    print(f"Raw embedding shape: {raw_embedding.shape}")

    # --- Test Contrastive Loss ---
    print("\n--- Testing Contrastive Loss ---")
    criterion = ContrastiveLoss(temperature=0.5)

    # Create two "views" (in reality these would be augmented versions)
    z1 = encoder(fake_data)
    z2 = encoder(fake_data + torch.randn_like(fake_data) * 0.1)  # Slightly different

    loss = criterion(z1, z2)
    print(f"Contrastive loss: {loss.item():.4f}")

    # --- Test Fall Classifier ---
    print("\n--- Testing Fall Detection Classifier ---")
    classifier = FallDetectionClassifier(embedding_dim=128, num_classes=2).to(device)

    # Get embeddings and classify
    with torch.no_grad():
        embeddings = encoder.get_embedding(fake_data)

    output = classifier(embeddings)
    predictions = torch.argmax(output, dim=1)

    print(f"Classifier output shape: {output.shape}")
    print(f"Predictions: {predictions[:10].tolist()}")
    print(f"  0 = Non-Fall, 1 = Fall")

    # --- Test backward pass (training works) ---
    print("\n--- Testing Backward Pass ---")
    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.001)

    z1 = encoder(fake_data)
    z2 = encoder(fake_data + torch.randn_like(fake_data) * 0.1)
    loss = criterion(z1, z2)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Backward pass successful ✅")
    print(f"Loss after 1 step: {loss.item():.4f}")

    print("\n✅ All model components working correctly!")
