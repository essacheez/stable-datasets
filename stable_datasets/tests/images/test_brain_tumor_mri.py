from collections import Counter

import pytest
from PIL import Image

from stable_datasets.images import BrainTumorMRI


IMAGES_PER_CLASS = {"train": 1400, "test": 400}


@pytest.mark.parametrize("split", ["train", "test"])
def test_brain_tumor_mri_dataset(split):
    dataset = BrainTumorMRI(split=split)

    images_per_class = IMAGES_PER_CLASS[split]
    assert len(dataset) == 4 * images_per_class

    sample = dataset[0]
    assert set(sample.keys()) == {"image", "label"}

    # Images keep their original size and color mode, so only check that they decode.
    image = sample["image"]
    assert isinstance(image, Image.Image)
    assert image.width > 0 and image.height > 0

    assert isinstance(sample["label"], int)
    label_counts = Counter(dataset[i]["label"] for i in range(len(dataset)))
    assert dict(label_counts) == dict.fromkeys(range(4), images_per_class)


def test_brain_tumor_mri_metadata():
    dataset = BrainTumorMRI(split="test")
    label_names = dataset.info.features["label"].names
    assert label_names == ["glioma", "meningioma", "notumor", "pituitary"]
