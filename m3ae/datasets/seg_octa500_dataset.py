import torch
from .base_dataset import BaseDataset


class OCTA500dDataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split

        if split == "train":
            names = ["seg_octa500_train"]
        elif split == "val":
            names = ["seg_octa500_val"]
        elif split == "test":
            names = ["seg_octa500_test"]
        else:
            raise ValueError

        super().__init__(*args, **kwargs, names=names, text_column_name="caption")
        # print(self.table["split"])
        # print(self.table["caption"])
        # print(self.table["image_id"])
        # self.label_column_name = self.label_column_name
        # print(self.table["image_id"].to_pandas().tolist())
        # self.labels = self.table["image_id"].to_pandas().tolist()
        # self.labels = self.table["image"].to_pandas().tolist()
        # assert len(self.labels) == len(self.table)

    def __getitem__(self, index):
        return self.get_suite(index)

    def get_suite(self, index):
        ret = super(OCTA500dDataset, self).get_suite(index)
        img_index, cap_index = self.index_mapper[index]
        # ret["image_id"] = self.labels[img_index][cap_index]

        # print(type(ret))
        # print(ret.keys())
        # print(len(ret['image']))
        # print(ret['image'][0])
        # print(0/0)
        # print(ret['raw_index'])
        # print(ret['text'])
        # print(ret['img_index'])
        # ret["seg_labels"] = self.labels[img_index][cap_index]
        return ret

    def collate(self, batch, mlm_collator):
        dict_batch = super(OCTA500dDataset, self).collate(batch, mlm_collator)
        # for sample in batch:
        #     print(sample["image"][0].shape)
        #     break

        # dict_batch["seg_labels"] = torch.tensor([sample["seg_labels"] for sample in batch])
        return dict_batch
