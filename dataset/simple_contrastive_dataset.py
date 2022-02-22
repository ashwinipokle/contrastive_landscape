from torch.utils.data import Dataset

class SimpleContrastiveDataset(Dataset):
    def __init__(self, data, Z, prob_ones=0.5):
        self.data = data
        self.Z = Z

        assert data.shape[0] == Z.shape[0]

        self.prob_ones = prob_ones

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        return x.astype(np.float), self.Z[idx].astype(np.int)