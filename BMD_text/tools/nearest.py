"""
Various nearest neighbour algorithms for dense word vectors
"""
import torch

class NearestNeighbours():
    def __init__(self, emb_array, device):
        """
        Computing cosine sim on vector and vocab every time is too costly. We
        can speed up computation by storing intermediates and change the order
        of the computation.
        """
        emb_array = torch.Tensor(emb_array).to(device)
        self.normalized_emb = self.normalize_matrix(emb_array)
        self.device = device

    def normalize_matrix(self, array):
        """
        Returns: array / euclidean_norm(array)
        """
        # Get embedding norm and normalize
        norm = torch.norm(array, dim=1)
        normed = array / norm.view(len(norm),1)
        return normed

    def normalize_tensor(self, array):
        """
        Returns: tensor / euclidean_norm(tensor), norm along 3rd dim
        """
        # Get embedding norm and normalize
        norm = torch.norm(array, dim=2)
        normed = array / norm.view(norm.shape[0], norm.shape[1], 1)
        return normed

    def cosine_sim(self, batch):
        """
        Compute consine efficiently. Assumes `batch is already euclidean
        normalized as well as embedding tensor
        """
        # Placeholder for results
        # Reshape to do a single matrix mult
        batch_size, seq_len = batch.shape[0], batch.shape[1]
        emb_size = self.normalized_emb.shape[1]
        batch = batch.view(-1, emb_size)
        mult = batch.matmul(self.normalized_emb.transpose(0,1))

        del batch # desperate attempt
        nearest = torch.argmax(mult, dim=1)
        nearest = nearest.view(batch_size, seq_len)
        return nearest

    def __call__(self, batch, mask=None):
        # Normalize batch. If normalize after dot product, mem explodes
        batch = self.normalize_tensor(batch)

        # Calc dist
        cos_sim = self.cosine_sim(batch)

        # Mask stuff
        if mask is not None:
            # Make sure types match
            cos_sim = cos_sim.int() * mask.int()

        return cos_sim

def old_nearest_neighbours(emb_array, batch, args, mask=None, near=1):
    """
    WARNING: depracated, and garbage

    Returns nearest neighbour embeddings in index form. If mask is provided, 0
    indices will be ignored.
    Args:
        emb_array = (np arr) all dataset embeddings
        batch: (np arr) batch to find nearesst neighbour from emb_array
        mask: (Tensor) same first 2-dim shape as `batch`. 0 elements ignored in
                `batch`
        """
    # Build list of distances and index in reference
    emb_array = torch.Tensor(emb_array).to(args.device)

    # OLD
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
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



