from .base_datamodule import BaseDataModule
from ..datasets import DRDBlindDataset


class DRDBlindDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return DRDBlindDataset

    @property
    def dataset_cls_no_false(self):
        return DRDBlindDataset

    @property
    def dataset_name(self):
        return "drd_blind"
