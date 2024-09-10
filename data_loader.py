import scipy.io


class BCIDatasetLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.train_data = None
        self.train_dg = None
        self.test_data = None
        self.load_data()

    def load_data(self):
        self.data = scipy.io.loadmat(self.file_path)

        self.train_data = self.data.get('train_data')  # ECoG signals (training)
        self.train_dg = self.data.get('train_dg')  # Finger positions (training)
        self.test_data = self.data.get('test_data')  # ECoG signals (testing)

        print(f"Train data shape: {self.train_data.shape}")
        print(f"Train finger data shape: {self.train_dg.shape}")
        print(f"Test data shape: {self.test_data.shape}")

    def get_train_data(self):
        return self.train_data, self.train_dg

    def get_test_data(self):
        return self.test_data

if __name__ == "__main__":
    file_path = 'data/sub1_comp.mat'
    loader = BCIDatasetLoader(file_path)

    train_data, train_dg = loader.get_train_data()
    test_data = loader.get_test_data()

    print("Training Data Loaded:", train_data.shape)
    print("Training Labels Loaded:", train_dg.shape)
    print("Test Data Loaded:", test_data.shape)
