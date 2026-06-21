import torch


def _normalize_with_self_loops(adj):
    """Add self loops and return D^(-1/2) A D^(-1/2)."""
    num_samples = adj.shape[0]
    eye = torch.eye(num_samples, device=adj.device, dtype=adj.dtype)
    adj = adj * (1.0 - eye) + eye
    row_sum = adj.sum(dim=-1, keepdim=True)
    d_inv_sqrt = torch.pow(row_sum + 1e-8, -0.5)
    return d_inv_sqrt * adj * d_inv_sqrt.transpose(-2, -1)


def _drop_non_self_edges(adj, probability):
    """Drop undirected non-self-loop edges and preserve symmetry."""
    if probability <= 0:
        return adj
    upper = torch.triu(adj, diagonal=1)
    keep = torch.rand_like(upper) >= probability
    upper = upper * keep
    return upper + upper.transpose(0, 1)


def build_sample_graph(
    features,
    k=5,
    mode="current",
    sim_threshold=0.5,
    dropedge=0.0,
    training=False,
):
    """Build a normalized sample graph from embeddings.

    ``current`` preserves the original directed top-k followed by max-symmetrization
    implementation. Mutual modes retain only reciprocal non-self neighbors. DropEdge
    is applied only to non-self edges before self loops and normalization.
    """
    valid_modes = {"current", "mutual_knn", "mutual_knn_threshold"}
    if mode not in valid_modes:
        raise ValueError(f"Unsupported graph mode: {mode}")

    num_samples = features.shape[0]
    cosine = torch.mm(
        torch.nn.functional.normalize(features, p=2, dim=1),
        torch.nn.functional.normalize(features, p=2, dim=1).t(),
    ).clamp(-1.0, 1.0)
    weights = (cosine + 1.0) / 2.0

    if mode == "current":
        # Exactly the previous graph rule: self can occur in the top-k set, then
        # symmetrization and an explicit self loop are applied below.
        _, indices = torch.topk(weights, min(k, num_samples), dim=-1)
        directed = torch.zeros_like(weights)
        directed.scatter_(-1, indices, 1.0)
        adj = torch.maximum(weights * directed, (weights * directed).t())
    else:
        # Mutual modes define k over non-self neighbors; self loops are added later.
        if num_samples <= 1:
            adj = torch.zeros_like(weights)
        else:
            neighbor_scores = cosine.clone()
            neighbor_scores.fill_diagonal_(float("-inf"))
            _, indices = torch.topk(neighbor_scores, min(k, num_samples - 1), dim=-1)
            directed = torch.zeros_like(weights)
            directed.scatter_(-1, indices, 1.0)
            mutual = directed * directed.t()
            if mode == "mutual_knn_threshold":
                mutual = mutual * (cosine > sim_threshold)
            adj = weights * mutual

    adj.fill_diagonal_(0.0)
    if training and dropedge > 0:
        adj = _drop_non_self_edges(adj, dropedge)
    return _normalize_with_self_loops(adj)
