"""
PyTorch Dataset for contrastive learning and evaluation
"""

import torch
from torch.utils.data import Dataset
import numpy as np

try:
    from src.augmentations import create_positive_pair
except ImportError:
    from augmentations import create_positive_pair


class ContrastiveIMUDataset(Dataset):
    """
    Dataset for contrastive learning training

    For each sample, it creates TWO augmented views
    The model learns: "These two views are the SAME motion"
    """

    def __init__(self, data, labels, sources):
        self.data = data.astype(np.float32)
        self.labels = labels
        self.sources = sources

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Create two augmented views of the same sample
        view1, view2 = create_positive_pair(sample)

        return (
            torch.tensor(view1, dtype=torch.float32),
            torch.tensor(view2, dtype=torch.float32),
            self.labels[idx],
            self.sources[idx],
        )


class IMUDataset(Dataset):
    """
    Standard dataset for evaluation and fall detection
    No augmentation, just returns the data as-is
    """

    def __init__(self, data, labels, sources):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = labels
        self.sources = sources

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.sources[idx]


# ============================================
# TEST
# ============================================

if __name__ == "__main__":
    print("Testing datasets...\n")

    # Create fake data
    fake_data = np.random.randn(100, 128, 3)
    fake_labels = np.array(["Walking"] * 50 + ["Fall"] * 50)
    fake_sources = np.array(["SisFall"] * 30 + ["WISDM"] * 40 + ["PAMAP2"] * 30)

    # Test ContrastiveIMUDataset
    print("--- ContrastiveIMUDataset ---")
    contrastive_ds = ContrastiveIMUDataset(fake_data, fake_labels, fake_sources)
    view1, view2, label, source = contrastive_ds[0]
    print(f"View 1 shape: {view1.shape}")
    print(f"View 2 shape: {view2.shape}")
    print(f"Label: {label}")
    print(f"Source: {source}")
    print(f"Views are different: {not torch.equal(view1, view2)}")

    # Test IMUDataset
    print("\n--- IMUDataset ---")
    standard_ds = IMUDataset(fake_data, fake_labels, fake_sources)
    data, label, source = standard_ds[0]
    print(f"Data shape: {data.shape}")
    print(f"Label: {label}")
    print(f"Source: {source}")

    # Test DataLoader
    print("\n--- DataLoader ---")
    from torch.utils.data import DataLoader

    loader = DataLoader(contrastive_ds, batch_size=16, shuffle=True)
    batch = next(iter(loader))
    v1, v2, labels, sources = batch
    print(f"Batch view1 shape: {v1.shape}")
    print(f"Batch view2 shape: {v2.shape}")
    print(f"Batch labels: {labels[:5]}")
    print(f"Batch sources: {sources[:5]}")

    print("\n✅ Datasets working correctly!")
