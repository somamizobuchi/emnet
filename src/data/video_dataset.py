from datasets import Dataset, np, Tuple, torch


class VideoDataset(Dataset):
    """
    A dataset class for loading and processing video data.

    Attributes:
        n_spatial (int): Spatial size of the crops.
        n_temporal (int): Temporal length of the crops.
    """

    def __init__(
        self,
        filename: str,
        n_spatial: int = 32,
        n_temporal: int = 96,
    ):
        """Spatial size of the crops."""
        self.n_spatial = n_spatial
        """Temporal length of the crops."""
        self.n_temporal = n_temporal

        # Extract length
        """The loaded pink noise video data."""
        self.data = np.load(filename, mmap_mode="r", allow_pickle=True)
        """The number of videos in the dataset."""
        self.length = self.data.shape[0]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index) -> torch.Tensor:
        """Randomly select a video index."""
        fi = torch.randint(self.data.shape[0], (1,))

        """Extract the crop and convert to float32."""
        coords = torch.randint(self.data.shape[2] - self.n_spatial, (2,))
        ti = torch.randint(self.data.shape[1] - self.n_temporal, (1,))
        out = (
            self.data[
                fi,
                ti : ti + self.n_temporal,
                coords[1] : coords[1] + self.n_spatial,
                coords[0] : coords[0] + self.n_spatial,
            ]
            .copy()
            .astype(np.float32)
        )
        """Normalize the crop."""
        out = (out - out.mean()) / out.std()

        return torch.from_numpy(out)
