from stable_datasets.utils import BaseDatasetBuilder


class BrainTumorMRI(BaseDatasetBuilder):
    """Brain Tumor MRI Dataset

    This dataset was put together by Masoud Nickparvar in 2021 by combining three older
    datasets (Figshare, SARTAJ and Br35H). It has 7,023 brain MRI images, and each image
    belongs to one of four classes: glioma, meningioma, pituitary tumor or no tumor.

    The images come in different sizes and are already split into a training set and a
    testing set.

    Still a work in progress. Downloading and loading the images will be added next.
    """

    @staticmethod
    def _labels():
        return ["glioma", "meningioma", "notumor", "pituitary"]
