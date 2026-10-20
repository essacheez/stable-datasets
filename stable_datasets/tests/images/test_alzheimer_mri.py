from collections import Counter

import numpy as np
import pytest
from PIL import Image

from stable_datasets.images import AlzheimerMRI


EXPECTED_LABEL_COUNTS = {
    "train": {0: 724, 1: 49, 2: 2566, 3: 1781},
    "test": {0: 172, 1: 15, 2: 634, 3: 459},
}


@pytest.mark.parametrize("split", ["train", "test"])
def test_alzheimer_mri_dataset(split):
    dataset = AlzheimerMRI(split=split)

    expected_counts = EXPECTED_LABEL_COUNTS[split]
    assert len(dataset) == sum(expected_counts.values())

    sample = dataset[0]
    assert set(sample.keys()) == {"image", "label"}

    image = sample["image"]
    assert isinstance(image, Image.Image)
    image_array = np.array(image)
    assert image_array.shape == (128, 128)
    assert image_array.dtype == np.uint8

    assert isinstance(sample["label"], int)
    label_counts = Counter(dataset[i]["label"] for i in range(len(dataset)))
    assert dict(label_counts) == expected_counts


def test_alzheimer_mri_metadata():
    dataset = AlzheimerMRI(split="test")
    label_names = dataset.info.features["label"].names
    assert label_names == ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]
