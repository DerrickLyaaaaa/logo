import numpy as np

from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors


try:
    import faiss  # type: ignore
    HAS_FAISS = True
except Exception:
    faiss = None
    HAS_FAISS = False


def _l2_normalize(x, eps=1e-12):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, eps, None)


def knn_search(features, k, return_distance=False):
    x = np.ascontiguousarray(features.astype(np.float32))
    n = x.shape[0]
    k = min(k, max(1, n - 1))

    if HAS_FAISS:
        index = faiss.IndexFlatL2(x.shape[1])
        index.add(x)
        d2, nn_idx = index.search(x, k + 1)
        if return_distance:
            dist = np.sqrt(np.clip(d2[:, 1:], 0.0, None)).astype(np.float32)
            return dist, nn_idx[:, 1:]
        return nn_idx[:, 1:]

    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nbrs.fit(x)
    if return_distance:
        dist, nn_idx = nbrs.kneighbors(x, return_distance=True)
        return dist[:, 1:].astype(np.float32), nn_idx[:, 1:]
    nn_idx = nbrs.kneighbors(x, return_distance=False)
    return nn_idx[:, 1:]


def knn_indices(features, k):
    return knn_search(features, k, return_distance=False)


def knn_adjacency(
    features,
    k=10,
    temp=0.07,
    mutual_only=False,
    local_scaling=False,
    local_scaling_k=0,
    return_stats=False,
):
    """
    Build a deterministic kNN graph from the original feature space.
    Returns row-normalized sparse adjacency W_tilde in CSR format.
    """
    x = _l2_normalize(features)
    n, _ = x.shape

    nn_idx = knn_indices(x, k=k)
    nn_sets = None
    if mutual_only:
        nn_sets = [set(row.tolist()) for row in nn_idx]

    edges = set()
    for i in range(n):
        for j in nn_idx[i]:
            j = int(j)
            if i == j:
                continue
            if mutual_only and i not in nn_sets[j]:
                continue
            a, b = (i, j) if i <= j else (j, i)
            edges.add((a, b))

    sigma = None
    if local_scaling and n > 1:
        k_sigma = int(local_scaling_k) if int(local_scaling_k) > 0 else int(k)
        k_sigma = min(max(1, k_sigma), max(1, n - 1))
        dist_knn, _ = knn_search(x, k=k_sigma, return_distance=True)
        sigma = np.clip(dist_knn[:, k_sigma - 1], 1e-6, None).astype(np.float32)

    rows = []
    cols = []
    vals = []
    node_sum = np.zeros(n, dtype=np.float32)
    node_cnt = np.zeros(n, dtype=np.float32)

    for i, j in edges:
        if sigma is not None:
            diff = x[i] - x[j]
            dist_sq = float(np.dot(diff, diff))
            denom = float(max(sigma[i] * sigma[j], 1e-12))
            w = float(np.exp(-dist_sq / denom))
        else:
            sim = float(np.dot(x[i], x[j]))
            w = float(np.exp(sim / max(temp, 1e-8)))

        rows.extend([i, j])
        cols.extend([j, i])
        vals.extend([w, w])
        node_sum[i] += w
        node_sum[j] += w
        node_cnt[i] += 1.0
        node_cnt[j] += 1.0

    rows.extend(list(range(n)))
    cols.extend(list(range(n)))
    vals.extend([1.0] * n)

    W = coo_matrix((np.asarray(vals, dtype=np.float32), (rows, cols)), shape=(n, n)).tocsr()
    row_sum = np.asarray(W.sum(axis=1)).reshape(-1)
    row_sum = np.clip(row_sum, 1e-12, None)
    inv = 1.0 / row_sum
    W_tilde = W.multiply(inv[:, None]).tocsr()

    if not return_stats:
        return W_tilde

    node_quality = node_sum / np.clip(node_cnt, 1.0, None)
    q_min = float(np.min(node_quality)) if node_quality.size else 0.0
    q_max = float(np.max(node_quality)) if node_quality.size else 1.0
    if q_max > q_min:
        edge_stability = ((node_quality - q_min) / (q_max - q_min)).astype(np.float32)
    else:
        edge_stability = np.ones(n, dtype=np.float32)

    return W_tilde, {
        "edge_stability": np.clip(edge_stability, 0.0, 1.0),
        "graph_edge_count": int(len(edges)),
    }


def stable_knn_adjacency(
    features,
    k=10,
    B=10,
    noise_std=0.01,
    stable_pi=0.6,
    temp=0.07,
    mutual_only=False,
    local_scaling=False,
    local_scaling_k=0,
    return_stats=False,
):
    """
    Build stable kNN graph from B noisy feature views.
    Returns row-normalized sparse adjacency W_tilde in CSR format.
    """
    x = _l2_normalize(features)
    n, _ = x.shape

    edge_counts = {}

    for _ in range(B):
        noisy = x + np.random.randn(*x.shape).astype(np.float32) * noise_std
        noisy = _l2_normalize(noisy)
        nn_idx = knn_indices(noisy, k=k)
        nn_sets = None
        if mutual_only:
            nn_sets = [set(row.tolist()) for row in nn_idx]

        for i in range(n):
            for j in nn_idx[i]:
                j = int(j)
                if mutual_only and i not in nn_sets[j]:
                    continue
                a, b = (i, int(j)) if i <= int(j) else (int(j), i)
                edge_counts[(a, b)] = edge_counts.get((a, b), 0) + 1

    stable_edges = []
    threshold = int(np.ceil(stable_pi * B))
    for (i, j), cnt in edge_counts.items():
        if cnt >= threshold and i != j:
            stable_edges.append((i, j))

    if not stable_edges:
        # Fallback to one-shot kNN if no edge survives.
        nn_idx = knn_indices(x, k=k)
        nn_sets = None
        if mutual_only:
            nn_sets = [set(row.tolist()) for row in nn_idx]
        fallback = set()
        for i in range(n):
            for j in nn_idx[i]:
                j = int(j)
                if i == j:
                    continue
                if mutual_only and i not in nn_sets[j]:
                    continue
                a, b = (i, j) if i <= j else (j, i)
                fallback.add((a, b))
        stable_edges = list(fallback)

    stable_edges = set(stable_edges)

    sigma = None
    if local_scaling and n > 1:
        k_sigma = int(local_scaling_k) if int(local_scaling_k) > 0 else int(k)
        k_sigma = min(max(1, k_sigma), max(1, n - 1))
        dist_knn, _ = knn_search(x, k=k_sigma, return_distance=True)
        sigma = np.clip(dist_knn[:, k_sigma - 1], 1e-6, None).astype(np.float32)

    rows = []
    cols = []
    vals = []

    for i, j in stable_edges:
        if sigma is not None:
            diff = x[i] - x[j]
            dist_sq = float(np.dot(diff, diff))
            denom = float(max(sigma[i] * sigma[j], 1e-12))
            w = float(np.exp(-dist_sq / denom))
        else:
            sim = float(np.dot(x[i], x[j]))
            w = float(np.exp(sim / max(temp, 1e-8)))

        rows.extend([i, j])
        cols.extend([j, i])
        vals.extend([w, w])

    # Add self-loops to avoid zero-degree rows.
    rows.extend(list(range(n)))
    cols.extend(list(range(n)))
    vals.extend([1.0] * n)

    W = coo_matrix((np.asarray(vals, dtype=np.float32), (rows, cols)), shape=(n, n)).tocsr()

    row_sum = np.asarray(W.sum(axis=1)).reshape(-1)
    row_sum = np.clip(row_sum, 1e-12, None)
    inv = 1.0 / row_sum

    # Row normalization: W_tilde = D^{-1} W
    W_tilde = W.multiply(inv[:, None]).tocsr()

    if not return_stats:
        return W_tilde

    # Node-wise edge stability: average edge occurrence frequency across noisy views.
    node_sum = np.zeros(n, dtype=np.float32)
    node_cnt = np.zeros(n, dtype=np.float32)
    denom_b = float(max(1, B))
    for (i, j), cnt in edge_counts.items():
        freq = float(cnt) / denom_b
        node_sum[i] += freq
        node_sum[j] += freq
        node_cnt[i] += 1.0
        node_cnt[j] += 1.0
    edge_stability = node_sum / np.clip(node_cnt, 1.0, None)
    edge_stability = np.clip(edge_stability, 0.0, 1.0).astype(np.float32)

    return W_tilde, {
        "edge_stability": edge_stability,
        "graph_edge_count": int(len(stable_edges)),
    }
