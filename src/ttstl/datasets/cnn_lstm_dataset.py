import numpy as np
from torch.utils.data import Dataset

class CNNLSTMDataset(Dataset):
    def __init__(self, inputs, target, window_size = 100):
        pad_inputs = np.zeros((window_size-1, inputs.shape[1]))
        self.inputs = inputs
        self.inputs = np.concatenate((pad_inputs, self.inputs), axis=0)
        
        self.target = target

        self.window_size = window_size

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return self.inputs[idx:idx + self.window_size], \
            self.target[idx]