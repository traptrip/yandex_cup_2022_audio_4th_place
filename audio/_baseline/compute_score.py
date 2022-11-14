# encoding=utf-8

import sys
import json
import numpy as np
import pandas as pd
from collections import defaultdict

def load_submission(input_path, max_top_size=100):
    result = {}
    with open(input_path, 'r') as finput:
        for line in finput:
            query_trackid, answer_items = line.rstrip().split('\t', 1)
            query_trackid = int(query_trackid)
            ranked_list = []
            for result_trackid in answer_items.split(' '):
                result_trackid = int(result_trackid)
                if result_trackid != query_trackid:
                    ranked_list.append(result_trackid)
                if len(ranked_list) >= max_top_size:
                    break
            result[query_trackid] = ranked_list
    return result

def position_discounter(position):
    return 1.0/np.log2(position+1)

def get_ideal_dcg(relevant_items_count, top_size):
    dcg = 0.0
    for result_indx in range(min(top_size, relevant_items_count)):
        position = result_indx + 1
        dcg += position_discounter(position)
    return dcg

def compute_dcg(query_trackid, ranked_list, track2artist_map):
    query_artistid = track2artist_map[query_trackid]
    dcg = 0.0
    for result_indx, result_trackid in enumerate(ranked_list):
        position = result_indx + 1
        discounted_position = position_discounter(position)
        result_artistid = track2artist_map[result_trackid]
        if result_artistid == query_artistid:
            dcg += discounted_position
    return dcg

def eval_submission(tracks_meta, submission, top_size):
    track2artist_map = {}
    artist2tracks_map = defaultdict(list)
    track2subset_map = {}
    for _, row in tracks_meta.iterrows():
        track2artist_map[row['trackid']] = row['artistid']
        track2subset_map[row['trackid']] = row['subset']
        artist2tracks_map[row['artistid']].append(row['trackid'])

    ndcg_list = defaultdict(list)
    for _, row in tracks_meta.iterrows():
        query_trackid = row['trackid']
        ranked_list = submission.get(query_trackid, [])
        query_artistid = track2artist_map[query_trackid]
        query_artist_tracks_count = len(artist2tracks_map[query_artistid])
        ideal_dcg = get_ideal_dcg(query_artist_tracks_count-1, top_size=top_size)
        dcg = compute_dcg(query_trackid, ranked_list, track2artist_map)
        ndcg_list[track2subset_map[query_trackid]].append(dcg/ideal_dcg)

    result = {}
    for subset, values in ndcg_list.items():
        result[subset.lower()] = np.mean(values)
    return result


if __name__ == '__main__':
    tracks_meta_path = sys.argv[1]
    submission_file_path = sys.argv[2]
    
    tracks_meta = pd.read_csv(tracks_meta_path, sep='\t')

    try:
        top_size = 100
        submission = load_submission(submission_file_path, max_top_size=top_size)
        scores = eval_submission(
            tracks_meta,
            submission,
            top_size=top_size
        )
        print(json.dumps(scores))
    except Exception as e:
        print("Error while reading answer file: " + str(e), file=sys.stderr)
        sys.exit(1)
