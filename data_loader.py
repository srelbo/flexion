import json
import pickle

import torch
from torch.utils.data import Dataset, DataLoader
import scipy.io
import numpy as np


# noinspection PyMethodMayBeStatic
class BCIDataset(Dataset):
    def __init__(self,
                 file_paths,
                 test_label_paths=None,
                 mode='train',
                 num_channels=48,
                 normalize=True,
                 train_stats=None):
        self.ecog_signals = []
        self.finger_flexions = []
        self.mode = mode
        self.num_channels = num_channels
        self.normalize = normalize

        self.train_stats = train_stats  # Training stats (mean, std for input, min/max for labels)

        if self.mode == 'train':
            # Load train data from the same files
            for file_path in file_paths:
                self.load_train_data(file_path)
        elif self.mode == 'test':
            if not test_label_paths:
                raise ValueError("Test labels file paths must be provided in 'test' mode.")
            # Load test data and corresponding labels
            for file_path, label_path in zip(file_paths, test_label_paths):
                self.load_test_data(file_path, label_path)

        # Stack the data from all subjects
        self.ecog_signals = torch.Tensor(np.vstack(self.ecog_signals))  # Stack all subject data

        if self.normalize:
            if self.mode == 'train':
                # Compute and store statistics for normalization
                self.mean, self.std = self.compute_stats(self.ecog_signals)
                self.ecog_signals = self.normalize_data(self.ecog_signals, self.mean, self.std)
                self.label_min, self.label_max = self.compute_label_min_max(self.finger_flexions)
                self.finger_flexions = self.normalize_labels(self.finger_flexions, self.label_min, self.label_max)
            else:
                # Use precomputed statistics during inference
                if self.train_stats is None:
                    raise ValueError("train_stats must be provided in test mode for normalization.")
                self.mean, self.std, self.label_min, self.label_max = self.train_stats
                self.ecog_signals = self.normalize_data(self.ecog_signals, self.mean, self.std)
                self.finger_flexions = self.normalize_labels(self.finger_flexions, self.label_min, self.label_max)

        if self.mode == 'train' or self.mode == 'test':
            self.finger_flexions = torch.Tensor(np.vstack(self.finger_flexions))  # Stack all finger data

    def load_train_data(self, file_path):
        """Load the train data and train_dg from the .mat file and append to the dataset"""
        data = scipy.io.loadmat(file_path)
        train_data = data['train_data']  # ECoG signals (training data)
        train_dg = data['train_dg']  # Finger flexions (training labels)

        # Adjust the channels if necessary
        if train_data.shape[1] != self.num_channels:
            train_data = self.adjust_channels(train_data)

        self.ecog_signals.append(train_data)
        self.finger_flexions.append(train_dg)

    def load_test_data(self, file_path, label_path):
        """Load the test data and corresponding labels"""
        data = scipy.io.loadmat(file_path)
        labels = scipy.io.loadmat(label_path)
        test_data = data['test_data']  # ECoG signals (testing data)
        test_labels = labels['test_dg']  # True finger flexions (test labels)

        # Adjust the channels if necessary
        if test_data.shape[1] != self.num_channels:
            test_data = self.adjust_channels(test_data)

        self.ecog_signals.append(test_data)
        self.finger_flexions.append(test_labels)

    def adjust_channels(self, data):
        """Pads or trims the data to have the correct number of channels."""
        current_channels = data.shape[1]
        if current_channels < self.num_channels:
            # Pad with zeros if fewer channels
            padding = np.zeros((data.shape[0], self.num_channels - current_channels))
            data = np.hstack([data, padding])
        else:
            # Trim extra channels if there are more
            data = data[:, :self.num_channels]
        return data

    def compute_stats(self, data):
        """Compute and return the mean and standard deviation per channel for Z-score normalization."""
        mean = torch.mean(data, dim=0)
        std = torch.std(data, dim=0)
        return mean, std

    def normalize_data(self, data, mean, std):
        """Apply Z-score normalization to the ECoG signals (across each channel)."""
        normalized_data = (data - mean) / std
        return normalized_data

    def compute_label_min_max(self, labels):
        """Compute the min and max per finger for Min-Max normalization."""
        labels = torch.Tensor(np.vstack(labels))  # Convert to tensor
        min_vals = torch.min(labels, dim=0).values
        max_vals = torch.max(labels, dim=0).values
        return min_vals, max_vals

    def normalize_labels(self, labels, min_vals, max_vals):
        """Apply Min-Max normalization to the finger flexion labels."""
        labels = torch.Tensor(np.vstack(labels))  # Convert to tensor
        normalized_labels = (labels - min_vals) / (max_vals - min_vals)  # Min-Max normalization
        return normalized_labels

    def __len__(self):
        return len(self.ecog_signals)

    def __getitem__(self, idx):
        return self.ecog_signals[idx], self.finger_flexions[idx]


if __name__ == "__main__":
    _train_file_paths = [
        'data/sub1_comp.mat',
        'data/sub2_comp.mat',
        'data/sub3_comp.mat'
    ]

    _test_file_paths = [
        'data/sub1_comp.mat',
        'data/sub2_comp.mat',
        'data/sub3_comp.mat'
    ]

    _test_label_paths = [
        'data/sub1_testlabels.mat',
        'data/sub2_testlabels.mat',
        'data/sub3_testlabels.mat'
    ]

    # Create training dataset
    train_dataset = BCIDataset(_train_file_paths, mode='train', num_channels=48, normalize=True)

    # Save training stats for inference
    _train_stats = (train_dataset.mean, train_dataset.std, train_dataset.label_min, train_dataset.label_max)
    print(f"Training stats: {_train_stats}")
    # Save the training stats to a JSON file
    with open('train_stats_v1.json', 'w') as f:
        # Convert to a dictionary
        train_stats_dict = {
            'mean': _train_stats[0].tolist(),
            'std': _train_stats[1].tolist(),
            'label_min': _train_stats[2].tolist(),
            'label_max': _train_stats[3].tolist()
        }
        json.dump(train_stats_dict, f)

    # Also save as pkl to preserve precision
    with open('train_stats_v1.pkl', 'wb') as f:
        pickle.dump(_train_stats, f)

    # Create test dataset and pass the saved training stats
    test_dataset = BCIDataset(_test_file_paths, test_label_paths=_test_label_paths, mode='test', num_channels=48,
                              normalize=True, train_stats=_train_stats)

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
