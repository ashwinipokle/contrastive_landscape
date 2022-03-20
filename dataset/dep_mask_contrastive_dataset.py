import numpy as np
from torch.utils.data import Dataset

class DependentMaskContrastiveDataset(Dataset):
    def __init__(self, data, Z):
        self.data = data
        self.Z = Z
        assert data.shape[0] == Z.shape[0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        p = x.shape[0]

        mask = np.random.choice([0, 1], (p, p), p=[0.5, 0.5])
        identity = np.eye(p)
        D = identity * mask

        a1 = 2 * np.matmul(D, x)
        a2 = 2 * np.matmul((np.identity(p) - D), x)

        assert np.sum(0.5 * (a1 + a2) - x) == 0, "Error in augmentation"
        return a1.astype(np.float), a2.astype(np.float), self.Z[idx].astype(np.int)