import os
import time
from pathlib import Path
from typing import Optional

import joblib
import faiss
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn import preprocessing


class BaseKNN(object):
    """KNN base class"""
    def __init__(self, embeddings: np.ndarray, ids: Optional[np.ndarray], method):
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        self.N = len(embeddings)
        self.D = embeddings[0].shape[-1]
        self.embeddings = embeddings if embeddings.flags['C_CONTIGUOUS'] \
                               else np.ascontiguousarray(embeddings)
        self.labels = ids

    def add(self, batch_size=10000):
        """Add data into index"""
        if self.N <= batch_size:
            self.index.add(self.embeddings)
        else:
            [self.index.add(self.embeddings[i:i+batch_size])
                    for i in range(0, len(self.embeddings), batch_size)]

    def search(self, queries, k=5):
        """Search
        Args:
            queries: query vectors
            k: get top-k results
        Returns:
            sims: similarities of k-NN
            ids: indexes of k-NN
        """
        if not queries.flags['C_CONTIGUOUS']:
            queries = np.ascontiguousarray(queries)
        if queries.dtype != np.float32:
            queries = queries.astype(np.float32)
        sims, ids = self.index.search(queries, k)
        ids = self.labels[ids] if self.labels is not None else ids
        return sims, ids


class KNN(BaseKNN):
    """KNN class
    Args:
        embeddings: feature vectors in database
        ids: labels of feature vectors
        method: distance metric
    """
    def __init__(self, embeddings: np.ndarray, ids: np.ndarray, method):
        super().__init__(embeddings, ids, method)
        self.index = {
            'cosine': faiss.IndexFlatIP,
            'euclidean': faiss.IndexFlatL2,
        }[method](self.D)
        if os.environ.get('CUDA_VISIBLE_DEVICES'):
            self.index = faiss.index_cpu_to_all_gpus(self.index)
        self.labels = ids
        self.add()


class ANN(BaseKNN):
    """Approximate nearest neighbor search class
    Args:
        embeddings: feature vectors in database
        ids: labels of feature vectors
        method: distance metric
    """
    def __init__(
        self, embeddings: np.ndarray, ids: Optional[np.ndarray], method="cosine", M=128, nbits=8, nlist=316, nprobe=64
    ):
        super().__init__(embeddings, ids, method)
        self.labels = ids
        self.quantizer = {
            'cosine': faiss.IndexFlatIP,
            'euclidean': faiss.IndexFlatL2
        }[method](self.D)
        self.index = faiss.IndexIVFPQ(self.quantizer, self.D, nlist, M, nbits)
        samples = embeddings[np.random.permutation(np.arange(self.N))[:self.N // 5]]
        self.index.train(samples)
        self.add()
        self.index.nprobe = nprobe


trunc_ids = None
trunc_init = None
lap_alpha = None


def get_offline_result(i):
    ids = trunc_ids[i]
    trunc_lap = lap_alpha[ids][:, ids]
    scores, _ = linalg.cg(trunc_lap, trunc_init, tol=1e-6, maxiter=20)
    return scores


def cache(filename):
    """Decorator to cache results"""

    def decorator(func):
        def wrapper(*args, **kw):
            self = args[0]
            path = os.path.join(self.cache_dir, filename)
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
            time0 = time.time()
            if os.path.exists(path):
                result = joblib.load(path)
                cost = time.time() - time0
                # print("[cache] loading {} costs {:.2f}s".format(path, cost))
                return result
            result = func(*args, **kw)
            cost = time.time() - time0
            print("[cache] obtaining {} costs {:.2f}s".format(path, cost))
            joblib.dump(result, path)
            return result

        return wrapper

    return decorator


class Diffusion(object):
    """Diffusion class"""

    def __init__(self, features, labels, cache_dir):
        self.features = features
        self.labels = np.array(labels)
        self.N = len(self.features)
        self.cache_dir = cache_dir
        # use ANN for large datasets
        self.use_ann = self.N >= 1_000_000
        if self.use_ann:
            print("ANN creating...")
            self.ann = ANN(self.features, None, method="cosine")
        self.knn = KNN(self.features, None, method="cosine")

    # @cache("offline.jbl")
    def get_offline_results(self, n_trunc, kd=50):
        """Get offline diffusion results for each gallery feature"""
        print("[offline] starting offline diffusion")
        print("[offline] 1) prepare Laplacian and initial state")
        global trunc_ids, trunc_init, lap_alpha
        if self.use_ann:
            _, trunc_ids = self.ann.search(self.features, n_trunc)
            sims, ids = self.knn.search(self.features, kd)
            lap_alpha = self.get_laplacian(sims, ids)
        else:
            sims, ids = self.knn.search(self.features, n_trunc)
            trunc_ids = ids
            lap_alpha = self.get_laplacian(sims[:, :kd], ids[:, :kd])
        trunc_init = np.zeros(n_trunc)
        trunc_init[0] = 1
        
        print("[offline] 2) gallery-side diffusion")
        results = Parallel(n_jobs=-1, prefer="threads")(
            delayed(get_offline_result)(i)
            for i in tqdm(range(self.N), desc="[offline] diffusion")
        )
        all_scores = np.concatenate(results)

        print("[offline] 3) merge offline results")
        rows = np.repeat(np.arange(self.N), n_trunc)

        offline = sparse.csr_matrix(
            (all_scores, (rows, trunc_ids.reshape(-1))),
            shape=(self.N, self.N),
            dtype=np.float32,
        )
        return offline

    # @cache('laplacian.jbl')
    def get_laplacian(self, sims, ids, alpha=0.99):
        """Get Laplacian_alpha matrix"""
        affinity = self.get_affinity(sims, ids)
        num = affinity.shape[0]
        degrees = affinity @ np.ones(num) + 1e-12
        # mat: degree matrix ^ (-1/2)
        mat = sparse.dia_matrix(
            (degrees ** (-0.5), [0]), shape=(num, num), dtype=np.float32
        )
        stochastic = mat @ affinity @ mat
        sparse_eye = sparse.dia_matrix(
            (np.ones(num), [0]), shape=(num, num), dtype=np.float32
        )
        lap_alpha = sparse_eye - alpha * stochastic
        return lap_alpha

    # @cache('affinity.jbl')
    def get_affinity(self, sims, ids, gamma=3):
        """Create affinity matrix for the mutual kNN graph of the whole dataset
        Args:
            sims: similarities of kNN
            ids: indexes of kNN
        Returns:
            affinity: affinity matrix
        """
        num = sims.shape[0]
        sims[sims < 0] = 0  # similarity should be non-negative
        sims = sims**gamma
        # vec_ids: feature vectors' ids
        # mut_ids: mutual (reciprocal) nearest neighbors' ids
        # mut_sims: similarites between feature vectors and their mutual nearest neighbors
        vec_ids, mut_ids, mut_sims = [], [], []
        for i in range(num):
            # check reciprocity: i is in j's kNN and j is in i's kNN when i != j
            ismutual = np.isin(ids[ids[i]], i).any(axis=1)
            ismutual[0] = False
            if ismutual.any():
                vec_ids.append(i * np.ones(ismutual.sum(), dtype=int))
                mut_ids.append(ids[i, ismutual])
                mut_sims.append(sims[i, ismutual])
        vec_ids, mut_ids, mut_sims = map(np.concatenate, [vec_ids, mut_ids, mut_sims])
        affinity = sparse.csc_matrix(
            (mut_sims, (vec_ids, mut_ids)), shape=(num, num), dtype=np.float32
        )
        return affinity


def offline_diffusion_search(
    queries,
    train_embeddings,
    labels,
    truncation_size=1000,
    kd=100,
    cache_dir="./cache",
):
    """
    Args:
        queries: predicted embeddings
        gallery: train embeddings
        cache_dir: Directory to cache embeddings
        truncation_size: Number of images in the truncated gallery
        kd: top k results
    """
    n_query = len(queries)
    diffusion = Diffusion(
        np.vstack([queries, train_embeddings]),
        labels,
        cache_dir=cache_dir,
    )
    offline = diffusion.get_offline_results(truncation_size, kd)
    features = preprocessing.normalize(offline, norm="l2", axis=1)
    scores = features[:n_query] @ features[n_query:].T

    ranks = np.argsort(-scores.toarray())[:, :kd]
    return ranks
