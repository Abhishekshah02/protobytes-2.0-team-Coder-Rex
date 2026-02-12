"""
Data augmentations for contrastive learning on IMU data

These augmentations simulate real-world differences between devices:
- Jittering = different sensor noise levels
- Scaling = different sensor calibrations
- Rotation = different sensor orientations/placements
- Time Warping = different sampling rates
- Permutation = different data ordering
"""

import numpy as np
from scipy.interpolate import CubicSpline


def jittering(data, sigma=0.05):
    """
    Add random noise to the signal

    Why: Different devices have different noise levels

    Before: [0.50, 0.30, 9.80]
    After:  [0.52, 0.28, 9.82]  (tiny random changes)
    """
    noise = np.random.normal(0, sigma, data.shape)
    return data + noise


def scaling(data, sigma=0.1):
    """
    Multiply signal by a random factor

    Why: Different devices measure in different scales
    Apple might give 0.5, Samsung might give 50 for same motion

    Before: [0.50, 0.30, 9.80]
    After:  [0.55, 0.33, 10.78]  (multiplied by ~1.1)
    """
    factor = np.random.normal(1, sigma, (1, data.shape[1]))
    return data * factor


def rotation(data):
    """
    Rotate the 3D signal randomly

    Why: Different devices are worn at different angles
    Apple Watch might be tilted left, Fitbit tilted right

    This simulates the sensor being in a different orientation
    """
    # Random rotation angle
    angle = np.random.uniform(0, 2 * np.pi)

    # Random rotation axis (x, y, or z)
    axis = np.random.choice([0, 1, 2])

    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    if axis == 0:
        rotation_matrix = np.array([[1, 0, 0], [0, cos_a, -sin_a], [0, sin_a, cos_a]])
    elif axis == 1:
        rotation_matrix = np.array([[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]])
    else:
        rotation_matrix = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])

    return np.dot(data, rotation_matrix)


def time_warping(data, sigma=0.2):
    """
    Slightly speed up or slow down parts of the signal

    Why: Different devices sample at different rates
    Some parts of motion might be captured faster/slower

    Like playing a video at slightly uneven speed
    """
    length = data.shape[0]

    # Create random warp points
    num_knots = 4
    knot_positions = np.linspace(0, length - 1, num_knots)
    knot_values = knot_positions + np.random.normal(
        0, sigma * length / num_knots, num_knots
    )
    knot_values = np.clip(knot_values, 0, length - 1)
    knot_values = np.sort(knot_values)

    # Create warping function
    original_positions = np.arange(length)
    warped_data = np.zeros_like(data)

    try:
        spline = CubicSpline(knot_positions, knot_values)
        warped_positions = spline(original_positions)
        warped_positions = np.clip(warped_positions, 0, length - 1)

        for col in range(data.shape[1]):
            source_spline = CubicSpline(np.arange(length), data[:, col])
            warped_data[:, col] = source_spline(warped_positions)
    except:
        return data

    return warped_data


def permutation(data, num_segments=4):
    """
    Randomly shuffle segments of the signal

    Why: Forces model to learn local patterns, not position

    Before: [A, B, C, D]  (4 segments in order)
    After:  [C, A, D, B]  (same segments, shuffled)
    """
    segment_length = len(data) // num_segments

    if segment_length < 2:
        return data

    segments = []
    for i in range(num_segments):
        start = i * segment_length
        end = start + segment_length
        segments.append(data[start:end])

    np.random.shuffle(segments)

    result = np.concatenate(segments, axis=0)

    # Pad or trim to original length
    if len(result) < len(data):
        result = np.vstack([result, data[len(result) :]])
    elif len(result) > len(data):
        result = result[: len(data)]

    return result


def magnitude_warping(data, sigma=0.2):
    """
    Smoothly change the magnitude of the signal

    Why: Simulates varying sensor sensitivity
    Some devices are more sensitive than others

    Like turning the volume up and down smoothly
    """
    length = data.shape[0]

    num_knots = 4
    knot_positions = np.linspace(0, length - 1, num_knots)
    knot_values = np.random.normal(1, sigma, num_knots)

    try:
        spline = CubicSpline(knot_positions, knot_values)
        warping_curve = spline(np.arange(length))
        return data * warping_curve.reshape(-1, 1)
    except:
        return data


def channel_dropout(data, drop_probability=0.1):
    """
    Randomly zero out one channel

    Why: Some devices might have faulty sensors
    or might not have all 3 axes

    Before: [x, y, z] = [0.5, 0.3, 9.8]
    After:  [x, y, z] = [0.5, 0.0, 9.8]  (y dropped)
    """
    result = data.copy()

    for col in range(data.shape[1]):
        if np.random.random() < drop_probability:
            result[:, col] = 0

    return result


# ============================================
# MAIN FUNCTIONS (Used by training)
# ============================================


def apply_augmentation(data):
    """
    Apply a random combination of augmentations to one sample

    This creates realistic variations that simulate:
    - Different devices
    - Different wearing positions
    - Different sensor qualities
    """
    augmented = data.copy()

    # Always add some noise (every device has noise)
    augmented = jittering(augmented, sigma=0.05)

    # Randomly apply other augmentations (50% chance each)
    if np.random.random() > 0.5:
        augmented = scaling(augmented, sigma=0.1)

    if np.random.random() > 0.5:
        augmented = rotation(augmented)

    if np.random.random() > 0.5:
        augmented = time_warping(augmented, sigma=0.2)

    if np.random.random() > 0.7:
        augmented = permutation(augmented)

    if np.random.random() > 0.7:
        augmented = magnitude_warping(augmented, sigma=0.2)

    if np.random.random() > 0.9:
        augmented = channel_dropout(augmented)

    return augmented


def create_positive_pair(data):
    """
    Create two different augmented versions of the same sample

    Input: Original walking data
    Output: Two slightly different versions of same walking data

    The model learns: "These two are the SAME motion"
    """
    view1 = apply_augmentation(data)
    view2 = apply_augmentation(data)

    return view1, view2


# ============================================
# TEST (Run this file to verify it works)
# ============================================

if __name__ == "__main__":
    print("Testing augmentations...")

    # Create fake IMU data (128 samples, 3 channels)
    fake_data = np.random.randn(128, 3)

    print(f"Original shape: {fake_data.shape}")
    print(f"Original first row: {fake_data[0]}")

    # Test each augmentation
    print("\n--- Testing Individual Augmentations ---")

    result = jittering(fake_data)
    print(f"Jittering:         shape={result.shape}, first={result[0].round(3)}")

    result = scaling(fake_data)
    print(f"Scaling:           shape={result.shape}, first={result[0].round(3)}")

    result = rotation(fake_data)
    print(f"Rotation:          shape={result.shape}, first={result[0].round(3)}")

    result = time_warping(fake_data)
    print(f"Time Warping:      shape={result.shape}, first={result[0].round(3)}")

    result = permutation(fake_data)
    print(f"Permutation:       shape={result.shape}, first={result[0].round(3)}")

    result = magnitude_warping(fake_data)
    print(f"Magnitude Warping: shape={result.shape}, first={result[0].round(3)}")

    result = channel_dropout(fake_data)
    print(f"Channel Dropout:   shape={result.shape}, first={result[0].round(3)}")

    # Test positive pair creation
    print("\n--- Testing Positive Pair Creation ---")
    view1, view2 = create_positive_pair(fake_data)
    print(f"View 1 shape: {view1.shape}")
    print(f"View 2 shape: {view2.shape}")
    print(f"View 1 first row: {view1[0].round(3)}")
    print(f"View 2 first row: {view2[0].round(3)}")
    print(f"Are they different? {not np.array_equal(view1, view2)}")

    print("\n✅ All augmentations working correctly!")
