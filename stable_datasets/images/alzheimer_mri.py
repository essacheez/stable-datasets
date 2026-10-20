from stable_datasets.utils import BaseDatasetBuilder


class AlzheimerMRI(BaseDatasetBuilder):
    """Alzheimer MRI Dataset

    This dataset has around 6,400 brain MRI slices. Each slice is a 2D grayscale image
    labeled with one of four stages of dementia: non demented, very mild, mild and moderate.

    It comes with its own train and test split, so no extra splitting is needed.

    Still a work in progress. Downloading and loading the images will be added next.
    """

    @staticmethod
    def _labels():
        return ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]
