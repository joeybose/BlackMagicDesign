"""
Various nearest neighbour algorithms for dense word vectors
"""
from scipy.spatial.distance import cosine
import torch

def nearest_neighbours(emb_array, batch, idx_to_tok, token_to_idx,args,near=1):
    """
    Returns nearest neighbour to batch in ids and string tokens
    Args:
        emb_array = (np arr) all dataset embeddings
        words: (np arr) batch to find nearesst neighbour from emb_array
        idx_to_tok: (list) return the string token given index where the index
                    matches to the embedding index
        token_to_idx: (dict) given string key, return index for prev args
    """
    # Build list of distances and index in reference
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    emb_array = torch.Tensor(emb_array).to(args.device)
    s = len(emb_array)
    nearest_tokens = []
    nearest_idx = torch.zeros((batch.shape[:2])).to(args.device)
    # Loop sequences
    for j in range(batch.shape[0]):
        seq = batch[0]
        # Find cosine between word and all word embeddings
        seq_tokens = []
        for i, word in enumerate(seq):
            neighbours = []
            # Get cosine
            w = word.unsqueeze(0).repeat(s, 1)
            score = cos(w, emb_array)
            max_score = score.argmax()
            nearest_idx[j, i] = max_score
            seq_tokens.append(idx_to_tok[max_score])
        seq_string = ' '.join(seq_tokens)
        nearest_tokens.append((seq_string))
    return nearest_idx, nearest_tokens
