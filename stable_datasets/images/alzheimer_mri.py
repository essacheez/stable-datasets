import pyarrow.parquet as pq

from stable_datasets.schema import ClassLabel, DatasetInfo, DatasetSource, DownloadInfo, Features, Version
from stable_datasets.schema import Image as ImageFeature
from stable_datasets.utils import BaseDatasetBuilder


HF_BASE_URL = (
    "https://huggingface.co/datasets/Falah/Alzheimer_MRI/resolve/daac24f9597236b45837d82f7eb9c9ad1f8c60c8/data"
)


class AlzheimerMRI(BaseDatasetBuilder):
    """Alzheimer MRI Dataset

    This dataset has 6,400 brain MRI slices. Each slice is a 128x128 grayscale image
    labeled with one of four stages of dementia: mild, moderate, non demented and very mild.

    It comes with its own train and test split (5,120 and 1,280 images), so no extra
    splitting is needed. The classes are very unbalanced. The moderate class only has
    49 training images while the non demented class has 2,566.
    """

    VERSION = Version("1.0.0")

    SOURCE = DatasetSource(
        homepage="https://huggingface.co/datasets/Falah/Alzheimer_MRI",
        assets={
            "train": DownloadInfo(url=f"{HF_BASE_URL}/train-00000-of-00001-c08a401c53fe5312.parquet"),
            "test": DownloadInfo(url=f"{HF_BASE_URL}/test-00000-of-00001-44110b9df98c5585.parquet"),
        },
        citation="""@misc{falah2023alzheimermri,
                      title={Alzheimer MRI Disease Classification Dataset},
                      author={Falah},
                      year={2023},
                      howpublished={\\url{https://huggingface.co/datasets/Falah/Alzheimer_MRI}}}""",
        license="Apache-2.0",
    )

    def _info(self):
        return DatasetInfo(
            description="Brain MRI slices labeled with one of four stages of Alzheimer's dementia.",
            features=Features(
                {
                    "image": ImageFeature(),
                    "label": ClassLabel(names=self._labels()),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            license=self.SOURCE["license"],
            citation=self.SOURCE["citation"],
        )

    def _generate_examples(self, data_path, split):
        """Read the parquet file and yield one image and its label per row."""
        table = pq.read_table(data_path)
        images = table.column("image").to_pylist()
        labels = table.column("label").to_pylist()

        for index, (image, label) in enumerate(zip(images, labels)):
            yield f"{split}_{index}", {"image": image["bytes"], "label": label}

    @staticmethod
    def _labels():
        # Same order as the label ids in the source parquet files (0 to 3).
        return ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]
