import torch

class TimeseriesDataset(torch.utils.data.Dataset):
    """
    Load data.
    Args:
        data: shape (num_h3_cells, num_time_steps)
        static_input_features: shape (num_h3_cells, features)
    """
    def __init__(
        self,
        data: torch.Tensor,
        tile_features: torch.Tensor,
        time_covariates: torch.Tensor,
        context_length: int,
        features: torch.Tensor | None = None, 
    ):
        self.data = data
        self.tile_features = tile_features
        self.time_covariates = time_covariates
        self.num_h3_cells, self.num_time_steps = data.shape
        self.features = features
        self.context_length = context_length

    def __len__(self):
        """
        Number of possible (batch_size, context_length) samples
        """
        # return (self.num_time_steps - self.context_length)
        return self.num_h3_cells * (self.num_time_steps - self.context_length)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get item.
        Returns:
            obs: observations of shape (num_h3, context_length, 1)
            targets: targets of shape (num_h3, context_length, 1)
        """
        h3_idx = self.get_h3_idx(idx)
        start = self.get_start_time_idx(idx)
        #obs = self.data[:, start:start+self.context_length]
        obs = self.data[h3_idx, start:start+self.context_length]

        #targets = self.data[:, start+1:start+self.context_length+1]
        targets = self.data[h3_idx, start+1:start+self.context_length+1]
        tile_features = self.tile_features[h3_idx].expand(obs.size(0), -1)
        time_covariates = self.time_covariates[start:start+self.context_length]

        obs = obs.unsqueeze(-1)
        targets = targets.unsqueeze(-1)

        features = None
        if self.features is not None:
            features = self.features[h3_idx, start:start+self.context_length]
            features = features.unsqueeze(-1)

        parts = [obs, tile_features, time_covariates]
        if features is not None:
            parts.append(features)
        obs = torch.concat(parts, dim=1)

        return obs, targets

    def get_h3_idx(self, idx: int) -> int:
        return idx // (self.num_time_steps - self.context_length)

    def get_start_time_idx(self, idx: int) -> int:
        return idx % (self.num_time_steps - self.context_length)

    def next_batch(self):
        pass
