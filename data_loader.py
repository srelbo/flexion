import torch
from torch.utils.data import Dataset, DataLoader
import scipy.io
import numpy as np

class BCIDataset(Dataset):
    def __init__(self, file_paths, test_label_paths=None, mode='train'):
        self.ecog_signals = []
        self.finger_flexions = []
        self.mode = mode

        if self.mode == 'train':
            for file_path in file_paths:
                self.load_train_data(file_path)
        elif self.mode == 'test':
            if not test_label_paths:
                raise ValueError("Test labels file paths must be provided in 'test' mode.")
            # Load test data and corresponding labels
            for file_path, label_path in zip(file_paths, test_label_paths):
                self.load_test_data(file_path, label_path)

        self.ecog_signals = torch.Tensor(np.vstack(self.ecog_signals))  # Stack all subject data
        if self.mode == 'train' or self.mode == 'test':
            self.finger_flexions = torch.Tensor(np.vstack(self.finger_flexions))  # Stack all finger data

    def load_train_data(self, file_path):
        data = scipy.io.loadmat(file_path)
        train_data = data['train_data']  # ECoG signals (training data)
        train_dg = data['train_dg']  # Finger flexions (training labels)

        self.ecog_signals.append(train_data)
        self.finger_flexions.append(train_dg)

    def load_test_data(self, file_path, label_path):
        data = scipy.io.loadmat(file_path)
        labels = scipy.io.loadmat(label_path)
        test_data = data['test_data']  # ECoG signals (testing data)
        test_labels = labels['test_dg']  # True finger flexions (test labels)

        self.ecog_signals.append(test_data)
        self.finger_flexions.append(test_labels)

    def __len__(self):
        return len(self.ecog_signals)

    def __getitem__(self, idx):
        if self.mode == 'train' or self.mode == 'test':
            return self.ecog_signals[idx], self.finger_flexions[idx]
        else:
            return self.ecog_signals[idx]


if __name__ == "__main__":
    _train_file_paths = [
        'data/sub1_comp.mat',
        'data/sub2_comp.mat',
        'data/sub3_comp.mat'
    ]

    _test_file_paths = [
        'data/sub1_test_data.mat',
        'data/sub2_test_data.mat',
        'data/sub3_test_data.mat'
    ]

    _test_label_paths = [
        'data/sub1_test_labels.mat',
        'data/sub2_test_labels.mat',
        'data/sub3_test_labels.mat'
    ]

    train_dataset = BCIDataset(_train_file_paths, mode='train')
    test_dataset = BCIDataset(_test_file_paths, test_label_paths=_test_label_paths, mode='test')

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    for batch_idx, (_ecog, _flexions) in enumerate(train_loader):
        print(f"Training Batch {batch_idx + 1}:")
        print(f"ECoG signal shape: {_ecog.shape}")
        print(f"Finger flexions shape: {_flexions.shape}")

    for batch_idx, (_ecog, _test_flexions) in enumerate(test_loader):
        print(f"Test Batch {batch_idx + 1}:")
        print(f"ECoG signal shape: {_ecog.shape}")
        print(f"Test flexions shape: {_test_flexions.shape}")
