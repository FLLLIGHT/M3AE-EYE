import json
import os
import random
import re
import csv

from make_arrow import make_arrow

def prepro_drd_blind():
    random.seed(42)

    # magic words
    text_dict = {
        '0': "This patient was diagnosed as no diabetic retinopathy.",
        '1': "This patient was diagnosed as mild diabetic retinopathy.",
        '2': "This patient was diagnosed as moderate diabetic retinopathy.",
        '3': "This patient was diagnosed as severe diabetic retinopathy.",
        '4': "This patient was diagnosed as proliferative diabetic retinopathy."
    }

    data = {
        "train": [],
        "val": [],
        "test": []
    }

    data_root = "data/pretrain_data/drd_blind"
    image_root = f"{data_root}/images/"

    drd_blind_samples = []
    with open(os.path.join(data_root, 'train.csv')) as sample_file:
        sample_file_reader = csv.reader(sample_file, delimiter=",")
        header = next(sample_file_reader)
        drd_blind_samples = list(sample_file_reader)
    indices = list(range(len(drd_blind_samples)))
    random.shuffle(indices)
    # 打乱并做数据集划分
    splits = {
        "train": indices[:-500],
        "val": indices[-500:-250],
        "test": indices[-250:],
    }
    for split, split_indices in splits.items():
        for sample_idx in split_indices:
            # 对于每一行，构建路径，比如：
            sample = drd_blind_samples[sample_idx]
            img_path = os.path.join(image_root, sample[0] + ".png")
            texts = [text_dict[sample[1]]]
            if len(texts) > 0:
                data[split].append({
                    "img_path": img_path,
                    "texts": texts
                })
    make_arrow(data, "drd_blind", "data/pretrain_arrows/")

def prepro_medicat(min_length=3):
    random.seed(42)

    data = {
        "train": [],
        "val": [],
        "test": []
    }

    data_root = "data/pretrain_data/medicat"
    image_root = f"{data_root}/release/figures/"
    medicat_ann_path = f"{data_root}/release/s2_full_figures_oa_nonroco_combined_medical_top4_public.jsonl"

    medicat_samples = [json.loads(sample) for sample in open(medicat_ann_path).read().strip().split("\n")]
    # 找出radiology相关的数据，每一行json都会有的数据是"radiology": true/false
    medicat_samples = [sample for sample in medicat_samples if sample["radiology"]]
    indices = list(range(len(medicat_samples)))
    random.shuffle(indices)
    # 打乱并做数据集划分
    splits = {
        "train": indices[:-2000],
        "val": indices[-2000:-1000],
        "test": indices[-1000:],
    }
    for split, split_indices in splits.items():
        for sample_idx in split_indices:
            # 对于每一行，构建路径，比如：
            # img_path: data/pretrain_data/medicat/release/figures/57c9ad0f4aab133f96d40992c46926fabc901ffa_2-Figure4-1.png
            sample = medicat_samples[sample_idx]
            img_path = os.path.join(image_root, sample["pdf_hash"] + "_" + sample["fig_uri"])
            texts = []
            # 图片的描述文本
            # eg: Figure 1. (A) Barium enema and (B) endoscopic image of the high-grade distal colonic obstruction caused
            if "s2_caption" in sample and len(sample["s2_caption"]) > 0:
                texts.append(sample["s2_caption"])
            
            # eg: "s2orc_references": ["Tissue hypertrophy and a small ulcer were noted at the site of the stricture (Figure 4 ).", "A pelvic radiograph showed that the stents remained partially in place, with absence of the most proximal and distal flares (Figure 4) ."]
            if "s2orc_references" in sample and sample["s2orc_references"] is not None and len(
                    sample["s2orc_references"]) > 0:
                texts.extend(sample["s2orc_references"])
            texts = [re.sub(r"\s+", " ", text.strip()) for text in texts]
            texts = [text for text in texts if len(text.split()) >= min_length]
            if len(texts) > 0:
                # 在对应的划分中加入数据（图像和对应的文本）
                data[split].append({
                    "img_path": img_path,
                    "texts": texts
                })

    make_arrow(data, "medicat", "data/pretrain_arrows/")


def prepro_roco(min_length=3):
    random.seed(42)

    data = {
        "train": [],
        "val": [],
        "test": []
    }
    roco_data_root = "data/pretrain_data/roco"
    roco_image_root = "data/pretrain_data/roco/{}/radiology/images/"
    medicat_roco_data_root = "data/pretrain_data/medicat"
    medicat_roco_paths = {
        "train": f"{medicat_roco_data_root}/net/nfs2.corp/allennlp/sanjays/roco_files/roco_train_references.jsonl",
        "val": f"{medicat_roco_data_root}/net/nfs2.corp/allennlp/sanjays/roco_files/roco_val_references.jsonl",
        "test": f"{medicat_roco_data_root}/net/nfs2.corp/allennlp/sanjays/roco_files/roco_test_references.jsonl"
    }

    medicat2roco = {}
    for split in ["train", "val", "test"]:
        with open(f"{roco_data_root}/{split}/radiology/dlinks.txt", "r") as fp:
            for line in fp:
                str_splits = line.strip().split('\t')
                # eg: medicat2roco[PMC4083729_AMHSR-4-14-g002.jpg] = "ROCO_00002"
                medicat2roco[str_splits[1].split(' ')[2].split('/')[-1].split('.')[0] + "_" + str_splits[-1]] = \
                str_splits[0]

    for split, path in medicat_roco_paths.items():
        samples = [json.loads(sample) for sample in open(path).read().strip().split("\n")]
        for sample in samples:
            img_path = os.path.join(roco_image_root.format(split), medicat2roco[sample["roco_image_id"]] + ".jpg")
            texts = []
            # 如果存在对应的文字，则加上
            if "gorc_references" in sample and sample["gorc_references"] is not None and len(
                    sample["gorc_references"]) > 0:
                texts.extend(sample["gorc_references"])
            texts = [re.sub(r"\s+", " ", text.strip()) for text in texts]
            texts = [text for text in texts if len(text.split()) >= min_length]
            if len(texts) > 0:
                data[split].append({
                    "img_path": img_path,
                    "texts": texts
                })

    for split in ["train", "val", "test"]:
        with open(f"{roco_data_root}/{split}/radiology/captions.txt", "r") as fp:
            for line in fp:
                str_splits = line.strip().split('\t')
                if len(str_splits) == 2:
                    # 注释：Computed tomography scan in axial view showing obliteration of the left maxillary sinus
                    img_path = os.path.join(roco_image_root.format(split), str_splits[0] + ".jpg")
                    texts = [str_splits[1]]
                    texts = [re.sub(r"\s+", " ", text.strip()) for text in texts]
                    texts = [text for text in texts if len(text.split()) >= min_length]
                    if len(texts) > 0:
                        data[split].append({
                            "img_path": img_path,
                            "texts": texts
                        })
    make_arrow(data, "roco", "data/pretrain_arrows/")


if __name__ == '__main__':
    prepro_drd_blind()
    # prepro_medicat()
    # prepro_roco()
