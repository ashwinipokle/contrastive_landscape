import numpy as np
from torch.utils.data import Dataset

class MaskedSparseContrastiveDataset(Dataset):
    def __init__(self, data, Z, prob_ones=0.5):
        self.data = data
        self.Z = Z

        assert data.shape[0] == Z.shape[0]

        self.prob_ones = prob_ones

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        p = x.shape[0]
        
        identity = np.eye(p)
        mask = np.random.choice([0, 1], (p, p), p=[1 - self.prob_ones, self.prob_ones])
        D1 = identity * mask

        mask = np.random.choice([0, 1], (p, p), p=[1 - self.prob_ones, self.prob_ones])
        D2 = identity * mask

        a1 = np.matmul(D1, x) # 2 * 
        a2 = np.matmul(D2, x) # 2 * 

        return a1.astype(np.float), a2.astype(np.float), self.Z[idx].astype(np.int)