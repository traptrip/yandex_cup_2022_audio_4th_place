# encoding=utf-8
import sys
sys.path.append("../")

import os
import random
from argparse import ArgumentParser

import annoy
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from models.net import RetrievalNet


class FeaturesLoader:
    def __init__(self, features_dir_path, meta_info, device="cpu", crop_size=60):
        self.features_dir_path = features_dir_path
        self.meta_info = meta_info
        self.trackid2path = meta_info.set_index("trackid")[
            "archive_features_path"
        ].to_dict()
        self.crop_size = crop_size
        self.device = device

    def _load_item(self, track_id):
        track_features_file_path = self.trackid2path[track_id]
        track_features = np.load(
            os.path.join(self.features_dir_path, track_features_file_path)
        )
        padding = (track_features.shape[1] - self.crop_size) // 2
        return track_features[:, padding : padding + self.crop_size]

    def load_batch(self, tracks_ids):
        batch = [self._load_item(track_id) for track_id in tracks_ids]
        return torch.tensor(np.array(batch)).to(self.device)


class TestLoader:
    def __init__(self, features_loader, batch_size=256, features_size=(512, 60)):
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


def get_ranked_list(embeds, top_size, annoy_num_trees=256):
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
        candidates = annoy_index.get_nns_by_item(id2annoy[track_id], top_size + 30)[1:]
        candidates = list(filter(lambda x: x != id2annoy[track_id], candidates))[:100]
        ranked_list[track_id] = [annoy2id[candidate] for candidate in candidates]
    return ranked_list


def inference(model, loader):
    embeds = dict()
    for tracks_ids, tracks_features in loader:
        with torch.no_grad():
            tracks_embeds = model(tracks_features)
            for track_id, track_embed in zip(tracks_ids, tracks_embeds):
                embeds[track_id] = track_embed.cpu().numpy()
    return embeds


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

    TESTSET_DIRNAME = "test_features"
    TRAINSET_DIRNAME = "train_features"
    TRAINSET_META_FILENAME = "train_meta_with_stages.tsv"
    TESTSET_META_FILENAME = "test_meta.tsv"
    SUBMISSION_FILENAME = "submission.txt"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    BATCH_SIZE = 800

    TRAINSET_PATH = os.path.join(args.base_dir, TRAINSET_DIRNAME)
    TESTSET_PATH = os.path.join(args.base_dir, TESTSET_DIRNAME)
    TRAINSET_META_PATH = os.path.join(args.base_dir, TRAINSET_META_FILENAME)
    TESTSET_META_PATH = os.path.join(args.base_dir, TESTSET_META_FILENAME)
    SUBMISSION_PATH = os.path.join(args.base_dir, SUBMISSION_FILENAME)
    CHECKPOINT_PATH = "../../experiments/audio/ROADMAP_AUDIO_512/weights/rolling.ckpt"

    model = RetrievalNet(
        "simCLR",
        256,
        norm_features=False,
        without_fc=True,
        with_autocast=False
    )
    model.load_state_dict(torch.load(CHECKPOINT_PATH)["net_state"])
    model.to(device)
    model.eval()

    meta_info = pd.read_csv(TRAINSET_META_PATH, sep="\t")
    test_meta_info = pd.read_csv(TESTSET_META_PATH, sep="\t")
    train_meta_info = meta_info[meta_info.stage == "train"].reset_index(drop=True)
    validation_meta_info = meta_info[meta_info.stage == "test"].reset_index(drop=True)

    print("Loaded data")
    print("Train set size: {}".format(len(train_meta_info)))
    print("Validation set size: {}".format(len(validation_meta_info)))
    print("Test set size: {}".format(len(test_meta_info)))
    print()

    print("Validation")
    test_loader = TestLoader(
        FeaturesLoader(TRAINSET_PATH, validation_meta_info, device), batch_size=BATCH_SIZE
    )
    embeds = inference(model, test_loader)
    submission = get_ranked_list(embeds, 100)
    score = eval_submission(submission, validation_meta_info)
    print(f"nDCG: {score}")

    print("Submission")
    test_loader = TestLoader(
        FeaturesLoader(TESTSET_PATH, test_meta_info, device), batch_size=BATCH_SIZE
    )
    embeds = inference(model, test_loader)
    submission = get_ranked_list(embeds, 100)
    save_submission(submission, SUBMISSION_PATH)


if __name__ == "__main__":
    main()
