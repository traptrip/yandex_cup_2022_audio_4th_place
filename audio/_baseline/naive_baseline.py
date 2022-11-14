# encoding=utf-8

import sys
import os
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from sklearn.metrics import pairwise_distances_chunked

def get_top_k(dist_chunk, start):
    top_size = 100
    result = []
    for chunk_item_indx, item_distances in enumerate(dist_chunk):
        global_query_item_indx = start + chunk_item_indx
        ranked_list = list(enumerate(item_distances))
        ranked_list.sort(key=lambda x: x[1])
        current_item_result = []
        for result_indx, distance in ranked_list:
            if result_indx == global_query_item_indx:
                continue
            current_item_result.append(result_indx)
            if len(current_item_result) >= top_size:
                break
        result.append(current_item_result)
    return result

if __name__ == '__main__':
    parser = ArgumentParser(description='Simple naive baseline')
    parser.add_argument('--features-dir', dest='local_features_dir', action='store', required=True)
    parser.add_argument('--tracks-meta', dest='test_tracks_meta_path', action='store', required=True)
    parser.add_argument('--output', dest='output_submission_path', action='store', required=True)
    args = parser.parse_args()

    test_tracks_meta = pd.read_csv(args.test_tracks_meta_path, sep='\t')
    trackids = []
    embeds = []

    for _, row in test_tracks_meta.iterrows():
        features_filepath = os.path.join(args.local_features_dir, row['archive_features_path'])
        track_features = np.load(features_filepath)
        track_embed = np.mean(track_features, axis=1)
        trackids.append(row['trackid'])
        embeds.append(track_embed)
    embeds = np.array(embeds)

    with open(args.output_submission_path, 'w') as foutput:
        current_item_indx = 0
        for chunk in pairwise_distances_chunked(embeds, metric='cosine', working_memory=100, reduce_func=get_top_k, n_jobs=16):
            for item_ranked_list in chunk:
                foutput.write('{}\t{}\n'.format(trackids[current_item_indx], ' '.join([str(trackids[i]) for i in item_ranked_list])))
                current_item_indx += 1
