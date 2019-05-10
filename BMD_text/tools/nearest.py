"""
Various nearest neighbour algorithms for dense word vectors
"""
import torch
from torch import nn
import ipdb
import torch.nn.functional as F

class NearestNeighbours(nn.Module):
    def __init__(self, emb_array, device, distance='L2'):
        """
        Computing cosine sim on vector and vocab every time is too costly. We
        can speed up computation by storing intermediates and change the order
        of the computation.
        """
        super().__init__()
        self.distance = distance
        # Normalize embedding (for faster cosine compute)
        emb_array = torch.Tensor(emb_array).to(device)
        if distance is not 'L2':
            normalized_emb = self.normalize_matrix(emb_array)
        else:
            normalized_emb = emb_array
        # Must make Parameter in order to parallelize
        self.normalized_emb = nn.Parameter(normalized_emb, requires_grad=False)
        self.register_parameter('embeddings', self.normalized_emb)
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

    def L2_distance(self, batch, return_all=False):
        """
        Compute consine efficiently. Assumes `batch is not euclidean
        normalized as well as embedding tensor

        Args:
            return_all: return L2 distance for all
        """
        batch_size, seq_len = batch.shape[0], batch.shape[1]
        emb_size = self.normalized_emb.shape[1]
        # Reshape to do a single matrix mult
        batch = batch.view(-1, emb_size)

    def cosine_distance(self, batch, return_all=False):
        """
        Compute consine efficiently. Assumes `batch is already euclidean
        normalized as well as embedding tensor

        Args:
            return_all: return cosine sim for all
        Cosine Similarity:
        <x, y> / ||x||*||y||
        """
        batch_size, seq_len = batch.shape[0], batch.shape[1]
        emb_size = self.normalized_emb.shape[1]
        # Reshape to do a single matrix mult
        batch = batch.view(-1, emb_size)
        # Cosine-sim's dot product
        mult = batch.matmul(self.normalized_emb.transpose(0,1))
        cosine_dist = 1 - mult

        del batch # desperate attempt to free memory
        nearest = torch.argmin(cosine_dist, dim=1)
        nearest = nearest.view(batch_size, seq_len)
        if return_all:
            return nearest, cosine_dist
        # Return only nearest to save memory
        return nearest

    def forward(self, batch, mask=None):
        # Normalize batch. If normalize after dot product, mem explodes
        batch = self.normalize_tensor(batch)

        # Calc dist
        if self.distance == 'L2':
            nearest = self.L2_distance(batch)
        else:
            nearest = self.cosine_distance(batch)

        # Mask stuff
        if mask is not None:
            # Make sure types match
            nearest = nearest.int() * mask.int()
        return nearest

class DiffNearestNeighbours(NearestNeighbours):
    def __init__(self, emb_array, device, diff_temp, decay_rate, distance, args):
        """
        Differentiable Nearest Neighbour. As temperature decreases, converges
        to nearest neighbour

        Args:
            nn_temp: softmax temp w/ nearest neighbour logits
        """
        super().__init__(emb_array, device, distance=distance)
        # self.temp = torch.tensor(diff_temp, requires_grad=False).to(device)
        # self.min_temp = torch.tensor(0.05, requires_grad=False).to(device)
        self.temp = diff_temp
        self.min_temp = 0.05
        self.batch_count = 0
        self.decay_rate = decay_rate
        self.args = args

    def temp_update(self):
        self.batch_count += 1
        if self.batch_count % self.decay_rate == 0:
            self.temp=torch.min(self.temp*0.95, self.min_temp)

    def forward(self, batch, mask=None, batch_token=None, test_temp=None):
        """
        Args:
            batch_token: batch but with original idx instead of embeddings, use
                this to compare the closest embedding indx, they should match
                if they should be the same word
            test_temp: will use this temp instead of self.temp, for debugging
        """
        # Maybe decay temp
        if test_temp is None:
            self.temp_update()
        else:
            self.temp = test_temp

        shape = batch.shape

        # Normalize batch. If normalize after dot product, mem explodes
        batch = self.normalize_tensor(batch)

        # Calc dist
        nearest, cosine_dists = self.cosine_distance(batch, return_all=True)

        # Mask stuff
        if mask is not None:
            # Make sure types match
            cosine_dists = cosine_dists.double() * mask.double()

        # Get weights to nearest neigh: Softmax with temp, reverse sign
        soft = F.softmax(((-1)*cosine_dists)/self.temp, dim=1)

        # Unit test
        # Most probable embedding must be same as nearest

        # New Embedding: Softmax over neighbours
        # emb = self.cheap_matmul(soft, self.normalized_emb)
        emb = soft.matmul(self.normalized_emb)

        # Reshape to original
        emb = emb.view(shape)

        return emb

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



