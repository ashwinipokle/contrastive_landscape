import numpy as np
from torch.utils.data import Dataset

class MultiMaskedSparseContrastiveDataset(Dataset):
    def __init__(self, data, Z, prob_ones=0.5, n_aug=5):
        self.data = data
        self.Z = Z

        self.n_aug = n_aug
        
        assert data.shape[0] == Z.shape[0]

        self.prob_ones = prob_ones

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        p = x.shape[0]

        a1_list = []
        a2_list = []

        for _ in range(self.n_aug):
            identity = np.eye(p)
            mask = np.random.choice([0, 1], (p, p), p=[1 - self.prob_ones, self.prob_ones])
            D1 = identity * mask

            mask = np.random.choice([0, 1], (p, p), p=[1 - self.prob_ones, self.prob_ones])
            D2 = identity * mask

            a1 = np.matmul(D1, x)
            a2 = np.matmul(D2, x)

            a1_list.append(a1.astype(np.float))
            a2_list.append(a2.astype(np.float))

        return a1_list, a2_list, self.Z[idx].astype(np.int)

# Custom collate for dataset
def multi_mask_data_collate(batch):
    all_a1 = []
    all_a2 = []
    all_z = []

    for a1_list, a2_list, z in batch:
        for a1, a2 in zip(a1_list, a2_list):
            all_a1.append(a1)
            all_a2.append(a2)
            all_z.append(z)

    all_a1 = torch.tensor(all_a1)
    all_a2 = torch.tensor(all_a2)
    all_z = torch.tensor(all_z)
    return all_a1, all_a2, all_z