import lmdb
import numpy as np
import os
import pandas as pd
import pickle
import random
import torch

from collections import namedtuple
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
import torchvision.transforms as T


# TODO: Update this dataset for use with unconditional models
CodeRow = namedtuple("CodeRow", ["top", "bottom", "label"])


class CUBDataset(Dataset):
    TRAIN_CODE = 0
    TEST_CODE = 1

    def __init__(
        self,
        root,
        tokenizer,
        mode="train",
        transform=None,
        max_length=64,
        return_label=False,
    ):
        if not os.path.isdir(root):
            raise ValueError(f"The specified root: {root} does not exist")
        self.root = root
        self.mode = mode
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.return_label = return_label

        self.images = []
        self.text_annotations = []

        image_mapping_path = os.path.join(self.root, "images.txt")
        split_mapping_path = os.path.join(self.root, "train_test_split.txt")
        cls_mapping_path = os.path.join(self.root, "image_class_labels.txt")
        cls_label_mapping_path = os.path.join(self.root, "classes.txt")

        img_path_map = pd.read_csv(
            image_mapping_path, header=None, index_col=0, sep=" "
        )
        index = img_path_map.index

        train_test_map = pd.read_csv(
            split_mapping_path, header=None, index_col=0, sep=" "
        )
        img_label_map = pd.read_csv(cls_mapping_path, header=None, index_col=0, sep=" ")
        label_cls_map = pd.read_csv(
            cls_label_mapping_path, header=None, index_col=0, sep=" "
        )
        img_cls_map = pd.Series(
            label_cls_map.loc[img_label_map[1]][1].to_numpy(), index=index
        )

        for img_path, cls_ in zip(img_path_map[1], img_cls_map):
            img_name = os.path.basename(os.path.splitext(img_path)[0])
            annotation_path = os.path.join(
                self.root, "text_c10", cls_, f"{img_name}.txt"
            )
            self.images.append(os.path.join(self.root, "images", img_path))
            self.text_annotations.append(annotation_path)

        self.images = pd.Series(self.images, index=index)
        self.text_annotations = pd.Series(self.text_annotations, index=index)

        # Select image subset based on the splitting mode
        if self.mode is not None:
            filter_code = self.TRAIN_CODE if self.mode == "train" else self.TEST_CODE
            valid_inds = index[train_test_map[1] == filter_code]
            self.images = self.images.loc[valid_inds]
            self.text_annotations = self.text_annotations.loc[valid_inds]

        # Remove grayscale images
        self.filtered_images = []
        self.filtered_annotations = []
        for img_path, annotation in tqdm(zip(self.images, self.text_annotations)):
            i = np.array(Image.open(img_path))
            if len(i.shape) == 2:
                continue
            self.filtered_images.append(img_path)
            self.filtered_annotations.append(annotation)

    def __getitem__(self, idx):
        img_path, annotation_path = (
            self.filtered_images[idx],
            self.filtered_annotations[idx],
        )
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)
        img_low_res = T.Resize((128, 128))(img)

        # Select one random annotation from the annotations file
        with open(annotation_path, "r") as fp:
            lines = fp.readlines()
            annotation = random.choice(lines)

        return img, img_low_res, annotation

    def collate_fn(self, batch):
        imgs, imgs_low_res, labels = zip(*batch)
        imgs_low_res = torch.stack(imgs_low_res, dim=0)
        imgs = torch.stack(imgs, dim=0)
        labels = list(labels)
        tokens = self.tokenizer(
            labels,
            padding=True,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        if self.return_label:
            return imgs, imgs_low_res, tokens, labels
        return imgs, imgs_low_res, tokens

    def __len__(self):
        assert len(self.filtered_images) == len(self.filtered_annotations)
        return len(self.filtered_images)


class LMDBDataset(Dataset):
    def __init__(self, path, tokenizer, max_length=64):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        self.tokenizer = tokenizer
        self.max_length = max_length
        if not self.env:
            raise IOError("Cannot open lmdb dataset", path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get("length".encode("utf-8")).decode("utf-8"))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = str(index).encode("utf-8")
            row = pickle.loads(txn.get(key))
        return torch.from_numpy(row.top), torch.from_numpy(row.bottom), row.label

    def collate_fn(self, batch):
        z_t, z_b, labels = zip(*batch)
        z_t = torch.stack(z_t, dim=0)
        z_b = torch.stack(z_b, dim=0)
        labels = list(labels)
        tokens = self.tokenizer(
            labels,
            padding=True,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return z_t, z_b, tokens
