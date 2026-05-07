import os
import numpy as np

from constants import TRAIN_FRACTION, SPLIT_RANDOM_SEED


def get_participant_id(path):
    """
    Example:
        SC4001E0-PSG.edf -> SC400
        SC4002E0-PSG.edf -> SC400

    Participants can have multiple nights of recordings. This will make sure that all nights from the same participant end up in the same dataset.
    """
    filename = os.path.basename(path)
    return filename[:5]


def split_paths_by_participant(psg_paths):
    """
    Split PSG paths into train and validation sets by participant.
    """

    psg_paths = sorted(psg_paths)
    # key = participant ID 
    # value = list of PSG files for that participant
    participants = {}

    for path in psg_paths:
        participant_id = get_participant_id(path)

        if participant_id not in participants:
            participants[participant_id] = []

        participants[participant_id].append(path)

    participant_ids = list(participants.keys())

    # Use a fixed random seed to make sure same split every time
    rng = np.random.default_rng(SPLIT_RANDOM_SEED)
    rng.shuffle(participant_ids)

    train_size = int(len(participant_ids) * TRAIN_FRACTION)

    if len(participant_ids) > 1:
        train_size = max(1, min(train_size, len(participant_ids) - 1))

    # If there is > 1 participant, make sure both sets get at least 1 participant.
    train_ids = participant_ids[:train_size]
    validation_ids = participant_ids[train_size:]

    train_paths = []
    validation_paths = []

    for participant_id in train_ids:
        train_paths.extend(participants[participant_id])

    for participant_id in validation_ids:
        validation_paths.extend(participants[participant_id])

    return sorted(train_paths), sorted(validation_paths)
