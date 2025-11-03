import torch
from torch.utils.data import Sampler
import numpy as np

class WeightedTileSampler(Sampler):
    """
    Weighted sampler for TimeseriesDataset.
    Each H3 cell is assigned a sampling weight, and time indices within
    the cell are sampled uniformly.
    
    Args:
        data_shape: tuple (num_h3_cells, num_time_steps)
        context_length: int
        h3_weights: tensor of shape (num_h3_cells,)
        num_samples: int (optional) number of samples per epoch
        replacement: bool (default=True)
    """
    def __init__(self, data_shape, context_length, h3_weights, num_samples=None, replacement=True):
        self.num_h3_cells, self.num_time_steps = data_shape
        self.context_length = context_length
        self.h3_weights = h3_weights / h3_weights.sum()  # normalize
        self.num_time_windows = self.num_time_steps - context_length
        self.num_samples = num_samples or (self.num_h3_cells * self.num_time_windows)
        self.replacement = replacement

        # Precompute per-sample weights for (cell, time_window)
        # Each time step within a cell gets the same weight
        weights_per_cell = self.h3_weights.repeat_interleave(self.num_time_windows)
        self.weights = weights_per_cell

    def __iter__(self):
        # Sample indices using torch.multinomial
        sampled_idxs = torch.multinomial(
            self.weights, 
            num_samples=self.num_samples, 
            replacement=self.replacement
        )
        return iter(sampled_idxs.tolist())

    def __len__(self):
        return self.num_samples
