"""
Various nearest neighbour algorithms for dense word vectors
"""
import torch

def nearest_neighbours(emb_array, batch, args, mask=None, near=1):
    """
    Returns nearest neighbour embeddings in index form. If mask is provided, 0
    indices will be ignored.
    Args:
        emb_array = (np arr) all dataset embeddings
        batch: (np arr) batch to find nearesst neighbour from emb_array
        mask: (Tensor) same first 2-dim shape as `batch`. 0 elements ignored in
                `batch`
        """
    # Build list of distances and index in reference
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    emb_array = torch.Tensor(emb_array).to(args.device)
    s = len(emb_array)
    nearest_idx = torch.zeros((batch.shape[:2])).to(args.device)
    # Loop sequences
    for j in range(batch.shape[0]):
        seq = batch[0]
        # Find cosine between word and all word embeddings
        seq_tokens = []
        for i, word in enumerate(seq):
            if mask is not None and float(mask[j,i]) == 0:
                nearest_idx[j, i] = 0
                continue
            neighbours = []
            # Get cosine
            w = word.unsqueeze(0).repeat(s, 1)
            score = cos(w, emb_array)
            max_score = score.argmax()
            nearest_idx[j, i] = max_score
    return nearest_idx
