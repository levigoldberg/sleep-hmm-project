import numpy as np

from constants import TRAIN_FRACTION, SPLIT_RANDOM_SEED


def split_paths_by_participant(psg_paths):
    """
    Split PSG files into train and validate sets by participant.
    """

    psg_paths = list(psg_paths)

    rng = np.random.default_rng(SPLIT_RANDOM_SEED)
    shuffled_indices = rng.permutation(len(psg_paths))

    train_size = int(len(psg_paths) * TRAIN_FRACTION)

    # Make sure both sets get at least one file when possible.
    if len(psg_paths) > 1:
        train_size = max(1, min(train_size, len(psg_paths) - 1))

    train_indices = shuffled_indices[:train_size]
    test_indices = shuffled_indices[train_size:]

    train_paths = [psg_paths[i] for i in train_indices]
    test_paths = [psg_paths[i] for i in test_indices]

    train_paths.sort()
    test_paths.sort()

    return train_paths, test_paths
