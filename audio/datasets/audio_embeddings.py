import os

import torch
import numpy as np
import pandas as pd
from sklearn import preprocessing

from .base_dataset import BaseDataset


class AudioEmbDataset(BaseDataset):
    def __init__(
        self,
        data_dir,
        csv_path,
        crop_size=60,
        mode="train",
        transform=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.get_fn = self._load_item

        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform
        self.crop_size = crop_size
        self.id_encoder = preprocessing.LabelEncoder()

        self.meta_info = pd.read_csv(csv_path, sep="\t")

        if self.mode in ["train", "val", "test", "all"]:
            self.meta_info["label"] = self.id_encoder.fit_transform(self.meta_info["artistid"])
            self.classes_ = self.id_encoder.classes_
            with open("classes.txt", "w") as f:
                f.write("\n".join(map(str, self.classes_)))
            if "stage" in self.meta_info.columns:
                self.meta_info = self.meta_info.loc[self.meta_info.stage == mode]

        self.paths = self.meta_info.archive_features_path.values
        self.labels = self.meta_info.label.values if mode != "submission" else None
        self.track_ids = self.meta_info.trackid.values

        self.super_labels = None
        if mode != "submission":
            self.get_instance_dict()
    
    def __process_features(self, x):
        x_len = x.shape[-1]
        if x_len > self.crop_size:
            start = np.random.randint(0, x_len - self.crop_size)
            x = x[..., start : start + self.crop_size]
        else:
            if self.mode == "train":
                i = np.random.randint(0, self.crop_size - x_len) if self.crop_size != x_len else 0
            else:
                i = (self.crop_size - x_len) // 2
            pad_patern = (i, self.crop_size - x_len - i)
            x = torch.nn.functional.pad(x, pad_patern, "constant").detach()
        # x /= x.max()
        x = (x - x.mean()) / x.std()
        return x


    def _load_item(self, idx):
        track_features_file_path = self.paths[idx]
        track_features = torch.from_numpy(np.load(
            os.path.join(self.data_dir, track_features_file_path)
        ))
        track_features = self.__process_features(track_features)

        if self.labels is not None:
            label = self.labels[idx]
            label = torch.tensor([label])

            out = {
                "features": track_features,
                "label": label,
                "path": track_features_file_path,
                "track_id": self.track_ids[idx],
            }
        else:
            out = {
                "features": track_features,
                "path": track_features_file_path,
                "track_id": self.track_ids[idx],
            }
            
        return out

    def __len__(self,):
        return len(self.paths)
