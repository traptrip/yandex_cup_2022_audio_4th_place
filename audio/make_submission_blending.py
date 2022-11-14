# encoding=utf-8
import os
import random
from argparse import ArgumentParser

import annoy
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from hydra import compose, initialize
from torch.utils.data import DataLoader

from models.net import RetrievalNet
from getter import Getter
from offline_diffusion import offline_diffusion_search, KNN


CROP_SIZE = 81
N_FOLDS = 10

def load_cfg(cfg_path):
    initialize(config_path=cfg_path, job_name="cfg")
    cfg = compose(
        config_name="default",
        overrides=[
            "experience.seed=42",
            "experience.data_dir=/home/and/projects/hacks/yandex_cup_2022/data/audio/train_features",
            "experience.experiment_name=EXP",
            "dataset.kwargs.csv_path=/home/and/projects/hacks/yandex_cup_2022/data/audio/train_meta_with_stages.tsv",
            f"dataset.kwargs.crop_size={CROP_SIZE}",
        ]
    )
    return cfg


getter = Getter()
CONFIG = load_cfg("./config")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAINSET_META_FILENAME = "train_meta_with_stages.tsv"
TESTSET_META_FILENAME = "test_meta.tsv"
SUBMISSION_FILENAME = "submission.txt"
CHECKPOINT_PATH = "../experiments/audio/folds_weights_f10"
BATCH_SIZE = 512


def get_ranked_list_diffusion(embeds, top_size=100):
    k = top_size + 1
    targets, embeddings = embeds
    ranks = offline_diffusion_search(embeddings, embeddings, None, truncation_size=k, kd=k)
    ranks = targets[ranks]

    ranked_list = dict()
    cnt = []
    for i, track_id in enumerate(targets):
        candidates = list(filter(lambda x: x != track_id, ranks[i, 1:]))
        ranked_list[track_id] = candidates[:100]
        cnt.append(len(ranked_list[track_id]))
    print(min(cnt), max(cnt), np.mean(cnt))

    return ranked_list


def inference(models, loader):
    all_embeddings = []
    all_track_ids = []

    for data in tqdm(loader):
        features = data["features"].to(DEVICE)
        track_ids = data["track_id"].tolist()

        embeddings = []
        for model in models:
            with torch.no_grad():
                embeddings.append(model(features))
        embeddings = torch.concat(embeddings, dim=-1)

        all_embeddings.append(embeddings.cpu())
        all_track_ids.extend(track_ids)

    all_embeddings = torch.concat(all_embeddings, dim=0)
    all_track_ids = np.array(all_track_ids)

    return all_track_ids, all_embeddings


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


def save_submission(submission, submission_path):
    with open(submission_path, "w") as f:
        for query_trackid, result in submission.items():
            f.write("{}\t{}\n".format(query_trackid, " ".join(map(str, result))))


class BasicNetV3(torch.nn.Module):
    def __init__(self, output_features_size=512):
        super().__init__()
        self.output_features_size = output_features_size
        self.transformer = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                output_features_size, 8, dim_feedforward=2048, dropout=0.25, batch_first=True
            ),
            num_layers=2
        )
        self.avg_pooling = torch.nn.AdaptiveAvgPool1d(1)


    def forward(self, x):
        x = self.transformer(x.float())
        # x = self.avg_pooling(x)
        return x


def main():
    parser = ArgumentParser(description="Validation & make submission")
    parser.add_argument("-b", "--base-dir", dest="base_dir", action="store", required=True)
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

    TRAINSET_META_PATH = os.path.join(args.base_dir, TRAINSET_META_FILENAME)
    TESTSET_META_PATH = os.path.join(args.base_dir, TESTSET_META_FILENAME)
    SUBMISSION_PATH = os.path.join(args.base_dir, SUBMISSION_FILENAME)

    # INIT MODELS
    models = []
    for i in range(N_FOLDS):
        weights_path = os.path.join(CHECKPOINT_PATH, f"fold{i}.ckpt")
        model = RetrievalNet(
            "simCLR",
            512,
            norm_features=True,
            without_fc=True,
            with_autocast=False
        )
        model.load_state_dict(torch.load(weights_path)["net_state"])    
        model.to(DEVICE)
        model.eval()
        models.append(model)

    model = RetrievalNet(
        "simCLR",
        512,
        norm_features=True,
        without_fc=True,
        with_autocast=False
    )
    model.load_state_dict(torch.load("../experiments/audio/F15_AUDIO_CROSSVAL_FOLD0_crop60_adamw_cosinelr_smclr_transformer_roadmap/weights/epoch_30.ckpt")["net_state"])    
    model.to(DEVICE)
    model.eval()
    models.append(model)
    
    #####

    meta_info = pd.read_csv(TRAINSET_META_PATH, sep="\t")
    test_meta_info = pd.read_csv(TESTSET_META_PATH, sep="\t")
    validation_meta_info = meta_info[meta_info.stage == "test"].reset_index(drop=True)

    print("Loaded data")
    print("Validation set size: {}".format(len(validation_meta_info)))
    print("Test set size: {}".format(len(test_meta_info)))
    print()

    transform = None
    print("Validation")
    CONFIG.dataset.kwargs.crop_size=81
    valid_ds = getter.get_dataset(transform, "test", CONFIG.dataset)
    valid_loader = DataLoader(
        valid_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    CONFIG.dataset.kwargs.crop_size=60
    valid_ds60 = getter.get_dataset(transform, "test", CONFIG.dataset)
    valid_loader60 = DataLoader(
        valid_ds60,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    embeds = inference(models, valid_loader, valid_loader60)
    submission = get_ranked_list_diffusion(embeds, 100)
    score = eval_submission(submission, validation_meta_info)
    print(f"nDCG: {score}")

    print("Submission")
    CONFIG.dataset.kwargs.data_dir = "./data/audio/test_features"
    CONFIG.dataset.kwargs.csv_path = "./data/audio/test_meta.tsv"
    CONFIG.dataset.kwargs.crop_size=81
    test_ds = getter.get_dataset(transform, "submission", CONFIG.dataset)
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    embeds = inference(models, test_loader)
    submission = get_ranked_list_diffusion(embeds, 100)
    save_submission(submission, SUBMISSION_PATH)


if __name__ == "__main__":
    main()
