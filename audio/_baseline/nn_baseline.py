# encoding=utf-8
import os
import random
from pathlib import Path
from argparse import ArgumentParser
from typing import Optional

import faiss
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from pytorch_metric_learning.losses import CentroidTripletLoss, TripletMarginLoss, NTXentLoss, ArcFaceLoss
from torchvision import models
from torch.cuda.amp import GradScaler, autocast

import annoy

DEVICE = "cuda"
# Data Loader
scaler = GradScaler()


def train_val_split(dataset, val_size=0.1):  # Сплит по artistid
    artist_ids = dataset["artistid"].unique()
    train_artist_ids, val_artist_ids = train_test_split(artist_ids, test_size=val_size)
    trainset = dataset[dataset["artistid"].isin(train_artist_ids)].copy()
    valset = dataset[dataset["artistid"].isin(val_artist_ids)].copy()
    return trainset, valset


class FeaturesLoader:
    def __init__(self, features_dir_path, meta_info, device="cpu", crop_size=81, mode="train"):
        self.features_dir_path = features_dir_path
        self.meta_info = meta_info
        self.trackid2path = meta_info.set_index("trackid")[
            "archive_features_path"
        ].to_dict()
        if mode != "test":
            self.trackid2label = meta_info.set_index("trackid")[
                "artistid"
            ].to_dict()
        self.crop_size = crop_size
        self.device = device
        self.mode = mode

    def __process_features(self, x):
        x = torch.from_numpy(x)
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
            x = torch.nn.functional.pad(x, pad_patern, "constant")
        return x.detach()

    def _load_item(self, track_id):
        track_features_file_path = self.trackid2path[track_id]
        track_features = np.load(
            os.path.join(self.features_dir_path, track_features_file_path)
        )
        track_features = self.__process_features(track_features)
        if self.mode != "test":
            label = self.trackid2label[track_id]
            return track_features, torch.tensor([label])
        else:
            return track_features, None

    def load_batch(self, tracks_ids):
        batch_features, batch_labels = [], []
        for track_id in tracks_ids:
            features, labels = self._load_item(track_id)
            batch_features.append(features[None])
            batch_labels.append(labels)
        
        if self.mode != "test":
            features_shape = batch_features[0].shape[-2:]
            batch_features = torch.cat(batch_features).reshape(-1, 1, *features_shape)
            labels = torch.cat(batch_labels)
            return batch_features, labels
        else:
            features_shape = batch_features[0].shape[-2:]
            batch_features = torch.cat(batch_features).reshape(-1, 1, *features_shape)
            return batch_features


class TrainLoader:
    def __init__(self, features_loader, batch_size=256, features_size=(1, 512, 81)):
        self.features_loader = features_loader
        self.batch_size = batch_size
        self.features_size = features_size
        self.artist_track_ids = self.features_loader.meta_info.groupby("artistid").agg(
            list
        )

    def _generate_pairs(self, track_ids):
        np.random.shuffle(track_ids)
        pairs = [track_ids[i - 2 : i] for i in range(2, len(track_ids) + 1, 2)]
        return pairs

    def _get_pair_ids(self):
        artist_track_ids = self.artist_track_ids.copy()
        artist_track_pairs = artist_track_ids["trackid"].map(self._generate_pairs)
        for pair_ids in artist_track_pairs.explode().dropna():
            yield pair_ids

    def _get_batch(self, batch_ids):
        batch_ids = np.array(batch_ids).reshape(-1)
        batch_features, batch_labels = self.features_loader.load_batch(batch_ids)
        batch_features = batch_features.reshape(self.batch_size, 2, *self.features_size)
        batch_labels = batch_labels.reshape(self.batch_size, 2)
        return batch_features, batch_labels

    def __iter__(self):
        batch_ids = []
        for pair_ids in self._get_pair_ids():
            batch_ids.append(pair_ids)
            if len(batch_ids) == self.batch_size:
                batch_features, batch_labels = self._get_batch(batch_ids)
                yield batch_features, batch_labels
                batch_ids = []


class TestLoader:
    def __init__(self, features_loader, batch_size=256, features_size=(512, 81)):
        self.features_loader = features_loader
        self.batch_size = batch_size
        self.features_size = features_size

    def __iter__(self):
        batch_ids = []
        for track_id in tqdm(self.features_loader.meta_info["trackid"].values):
            batch_ids.append(track_id)
            if len(batch_ids) == self.batch_size:
                yield batch_ids, self.features_loader.load_batch(batch_ids)
                batch_ids = []
        if len(batch_ids) > 0:
            yield batch_ids, self.features_loader.load_batch(batch_ids)


# Loss & Metrics


class NT_Xent(nn.Module):
    def __init__(self, temperature):
        super(NT_Xent, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        batch_size = z_i.shape[0]
        N = 2 * batch_size
        z = torch.cat((z_i, z_j), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        mask = self.mask_correlated_samples(batch_size)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        with torch.no_grad():
            top1_negative_samples, _ = negative_samples.topk(1)
            avg_rank = (
                logits.argsort(descending=True)
                .argmin(dim=1)
                .float()
                .mean()
                .cpu()
                .numpy()
            )

        return loss, avg_rank
    

def get_ranked_list(embeds, top_size, annoy_num_trees=32):
    annoy_index = None
    annoy2id = []
    id2annoy = dict()

    for track_id, track_embed in embeds.items():
        id2annoy[track_id] = len(annoy2id)
        annoy2id.append(track_id)
        if annoy_index is None:
            annoy_index = annoy.AnnoyIndex(len(track_embed), "angular")
        annoy_index.add_item(id2annoy[track_id], track_embed)
    annoy_index.build(annoy_num_trees)

    ranked_list = dict()
    for track_id in embeds.keys():
        candidates = annoy_index.get_nns_by_item(id2annoy[track_id], top_size)[
            1:
        ]  # exclude trackid itself
        candidates = list(filter(lambda x: x != id2annoy[track_id], candidates))
        ranked_list[track_id] = [annoy2id[candidate] for candidate in candidates]
    return ranked_list


def position_discounter(position):
    return 1.0 / np.log2(position + 1)

def get_ideal_dcg(relevant_items_count, top_size):
    dcg = 0.0
    for result_indx in range(min(top_size, relevant_items_count)):
        position = result_indx + 1
        dcg += position_discounter(position)
    return dcg

def compute_dcg(query_trackid, ranked_list, track2artist_map, top_size):
    query_artistid = track2artist_map[query_trackid]
    dcg = 0.0
    for result_indx, result_trackid in enumerate(ranked_list[:top_size]):
        assert result_trackid != query_trackid
        position = result_indx + 1
        discounted_position = position_discounter(position)
        result_artistid = track2artist_map[result_trackid]
        if result_artistid == query_artistid:
            dcg += discounted_position
    return dcg

# def dcg(knn_labels, query_labels, k=100):
#     for knn_label, query_label in zip(knn_labels, query_labels):
#         ideal_dcg = 
#         dcg = 0.0
#         for i, pred_id in enumerate(knn_label):
#             position = i + 1
#             discounted_position = position_discounter(position)
#             if pred_id == query_label:
#                 dcg += discounted_position
        

def eval_submission(submission, gt_meta_info, top_size=100):
    track2artist_map = gt_meta_info.set_index("trackid")["artistid"].to_dict()
    artist2tracks_map = gt_meta_info.groupby("artistid").agg(list)["trackid"].to_dict()
    ndcg_list = []
    for query_trackid in tqdm(submission.keys()):
        ranked_list = submission[query_trackid]
        query_artistid = track2artist_map[query_trackid]
        query_artist_tracks_count = len(artist2tracks_map[query_artistid])
        ideal_dcg = get_ideal_dcg(query_artist_tracks_count - 1, top_size=top_size)
        dcg = compute_dcg(
            query_trackid, ranked_list, track2artist_map, top_size=top_size
        )
        try:
            ndcg_list.append(dcg / ideal_dcg)
        except ZeroDivisionError:
            continue
    return np.mean(ndcg_list)


# Train & Inference functions


class BasicNet(nn.Module):
    def __init__(self, output_features_size):
        super().__init__()
        self.output_features_size = output_features_size
        self.conv_1 = nn.Conv1d(512, output_features_size, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv1d(
            output_features_size, output_features_size, kernel_size=3, padding=1
        )
        self.mp_1 = nn.MaxPool1d(2, 2)
        self.conv_3 = nn.Conv1d(
            output_features_size, output_features_size, kernel_size=3, padding=1
        )
        self.conv_4 = nn.Conv1d(
            output_features_size, output_features_size, kernel_size=3, padding=1
        )

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = F.relu(self.conv_2(x))
        x = self.mp_1(x)
        x = F.relu(self.conv_3(x))
        x = self.conv_4(x).mean(axis=2)
        return x


class SimCLR(nn.Module):
    def __init__(self, enc_dim, projection_dim):
        super().__init__()
        # self.encoder = encoder
        encoder = models.resnet18(pretrained=True)
        encoder.conv1 = nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        encoder.fc = torch.nn.Linear(encoder.fc.in_features, enc_dim)
        self.encoder = encoder

        self.n_features = enc_dim
        self.projection_dim = projection_dim
        self.projector = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.ReLU(),
            nn.Linear(self.n_features, self.projection_dim, bias=False),
        )

    def forward(self, x_i, x_j):
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)

        z_i = self.projector(h_i)
        z_j = self.projector(h_j)
        return h_i, h_j, z_i, z_j
 

def inference(model, loader):
    embeds = dict()
    with torch.no_grad(), autocast():
        for tracks_ids, tracks_features in loader:
            tracks_features = tracks_features.to(DEVICE)
            tracks_embeds = model(tracks_features)
            for track_id, track_embed in zip(tracks_ids, tracks_embeds):
                embeds[track_id] = track_embed.cpu().numpy()
    return embeds


def train(
    module,
    train_loader,
    val_loader,
    valset_meta,
    optimizer,
    criterion,
    num_epochs,
    checkpoint_path,
    top_size=100,
):
    try:
        max_ndcg = None
        for epoch in range(num_epochs):
            print("\nEpoch {}/{}".format(epoch + 1, num_epochs))
            module.train()
            for batch in train_loader:
                optimizer.zero_grad()

                with autocast():
                    batch_features, batch_labels = batch
                    batch_features, batch_labels = batch_features.to(DEVICE), batch_labels.to(DEVICE)
                    x_i, x_j = batch_features[:, 0, :, :], batch_features[:, 1, :, :]
                    labels, ref_labels = batch_labels[:, 0].squeeze(), batch_labels[:, 1].squeeze()

                    # print(x_i.shape)
                    h_i, h_j, z_i, z_j = module(x_i, x_j)
                    loss, _ = criterion(z_i, z_j)
                    # loss = criterion(
                    #     z_i, labels=labels, ref_emb=z_j, ref_labels=ref_labels
                    # )
                    # loss = criterion(
                    #     z_i, labels=labels
                    # )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # loss.backward()
                # optimizer.step()
            # print(f"loss: {loss} avg_rank: {avg_rank}".format(loss))
            print(f"loss: {loss}")
            print()

            module.eval()

            model_encoder = module.encoder
            embeds_encoder = inference(model_encoder, val_loader)
            ranked_list_encoder = get_ranked_list(embeds_encoder, top_size)
            val_ndcg_encoder = eval_submission(ranked_list_encoder, valset_meta)

            model_projector = nn.Sequential(module.encoder, module.projector)
            embeds_projector = inference(model_projector, val_loader)
            ranked_list_projector = get_ranked_list(embeds_projector, top_size)
            val_ndcg_projector = eval_submission(ranked_list_projector, valset_meta)

            print("Validation nDCG on epoch {}".format(epoch))
            print("Encoder - {}".format(val_ndcg_encoder))
            print("Projector - {}".format(val_ndcg_projector))
            if (max_ndcg is None) or (val_ndcg_encoder > max_ndcg):
                max_ndcg = val_ndcg_encoder
                torch.save(model_encoder.state_dict(), checkpoint_path)
                torch.save(model_projector.state_dict(), checkpoint_path.replace("best", "best_projector"))
    except KeyboardInterrupt:
        pass

def save_submission(submission, submission_path):
    with open(submission_path, "w") as f:
        for query_trackid, result in submission.items():
            f.write("{}\t{}\n".format(query_trackid, " ".join(map(str, result))))


def main():
    parser = ArgumentParser(description="Simple naive baseline")
    parser.add_argument("--base-dir", dest="base_dir", action="store", required=True)
    args = parser.parse_args()

    # Seed
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    TRAINSET_DIRNAME = "train_features"
    TESTSET_DIRNAME = "test_features"
    TRAINSET_META_FILENAME = "train_meta_balanced_with_stages.tsv"
    TESTSET_META_FILENAME = "test_meta.tsv"
    SUBMISSION_FILENAME = "submission.txt"
    MODEL_FILENAME = "model.pt"
    CHECKPOINT_FILENAME = "best.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    BATCH_SIZE = 256
    N_CHANNELS = 256
    PROJECTION_DIM = 128
    NUM_EPOCHS = 30
    LR = 1e-4
    TEMPERATURE = 0.1
    CROP_SIZE = 81

    TRAINSET_PATH = os.path.join(args.base_dir, TRAINSET_DIRNAME)
    TESTSET_PATH = os.path.join(args.base_dir, TESTSET_DIRNAME)
    TRAINSET_META_PATH = os.path.join(args.base_dir, TRAINSET_META_FILENAME)
    TESTSET_META_PATH = os.path.join(args.base_dir, TESTSET_META_FILENAME)
    SUBMISSION_PATH = os.path.join(args.base_dir, SUBMISSION_FILENAME)
    MODEL_PATH = os.path.join(args.base_dir, MODEL_FILENAME)
    CHECKPOINT_PATH = os.path.join(args.base_dir, CHECKPOINT_FILENAME)

    sim_clr = SimCLR(enc_dim=N_CHANNELS, projection_dim=PROJECTION_DIM)
    sim_clr.to(device)

    meta_info = pd.read_csv(TRAINSET_META_PATH, sep="\t")
    meta_info["artistid"] = LabelEncoder().fit_transform(meta_info["artistid"])
    N_CLASSES = meta_info["artistid"].nunique()
    test_meta_info = pd.read_csv(TESTSET_META_PATH, sep="\t")
    # train_meta_info, validation_meta_info = train_val_split(
    #     train_meta_info, val_size=0.1
    # )
    train_meta_info = meta_info[meta_info.stage == "train"].reset_index(drop=True)
    validation_meta_info = meta_info[meta_info.stage == "test"].reset_index(drop=True)

    print("Loaded data")
    print("Train set size: {}".format(len(train_meta_info)))
    print("Validation set size: {}".format(len(validation_meta_info)))
    print("Test set size: {}".format(len(test_meta_info)))
    print()

    print("Train")
    train(
        module=sim_clr,
        train_loader=TrainLoader(
            FeaturesLoader(TRAINSET_PATH, train_meta_info, device, crop_size=CROP_SIZE, mode="train"),
            batch_size=BATCH_SIZE,
            features_size=(1, 512, CROP_SIZE),
        ),
        val_loader=TestLoader(
            FeaturesLoader(TRAINSET_PATH, validation_meta_info, device, crop_size=CROP_SIZE, mode="test"),
            batch_size=BATCH_SIZE,
            features_size=(1, 512, CROP_SIZE),
        ),
        valset_meta=validation_meta_info,
        optimizer=torch.optim.Adam(sim_clr.parameters(), lr=LR),
        criterion=NT_Xent(temperature=TEMPERATURE), # ArcFaceLoss(num_classes=N_CLASSES, embedding_size=PROJECTION_DIM, margin=28.6, scale=64), # NTXentLoss(TEMPERATURE), # TripletMarginLoss(margin=0.05, swap=False, smooth_loss=False), # NT_Xent(temperature=TEMPERATURE), # TripletMarginLoss(margin=0.05, swap=False, smooth_loss=False), # 
        num_epochs=NUM_EPOCHS,
        checkpoint_path=CHECKPOINT_PATH,
    )

    print("Submission")
    test_loader = TestLoader(
        FeaturesLoader(TESTSET_PATH, test_meta_info, device, crop_size=CROP_SIZE, mode="test"), 
        batch_size=BATCH_SIZE,
        features_size=(1, 512, CROP_SIZE),
    )
    model = sim_clr.encoder
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    model.to(device)
    model.eval()
    embeds = inference(model, test_loader)
    submission = get_ranked_list(embeds, 100)
    # submission = get_knn(embeds, k=100)
    save_submission(submission, SUBMISSION_PATH)
    torch.save(sim_clr.state_dict(), MODEL_PATH)


if __name__ == "__main__":
    main()
