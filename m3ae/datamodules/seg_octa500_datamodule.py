from .base_datamodule import BaseDataModule
from ..datasets import OCTA500dDataset


class OCTA500DataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return OCTA500dDataset

    @property
    def dataset_cls_no_false(self):
        return OCTA500dDataset

    @property
    def dataset_name(self):
        return "seg_octa500"
