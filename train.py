import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb  # Add this line to import wandb

from data_loader import BCIDataset
from models.unet import UNet1D


class Trainer:
    def __init__(self, model, train_loader, test_loader, device=None, learning_rate=0.001, num_epochs=50):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        self.criterion = nn.MSELoss()  # Mean Squared Error loss for regression
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Initialize WandB
        wandb.init(project="flexion", config={
            "learning_rate": self.learning_rate,
            "epochs": self.num_epochs,
            "batch_size": len(self.train_loader.dataset)
        })
        wandb.watch(self.model, log="all")

    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0

            for batch_idx, (ecog_signals, finger_flexions) in enumerate(self.train_loader):
                ecog_signals, finger_flexions = ecog_signals.to(self.device), finger_flexions.to(self.device)
                ecog_signals = ecog_signals.unsqueeze(2)  # Add channel dimension
                finger_flexions.unsqueeze(2)
                outputs = self.model(ecog_signals)
                loss = self.criterion(outputs, finger_flexions)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(self.train_loader)
            print(f"Epoch [{epoch + 1}/{self.num_epochs}], Loss: {avg_loss}")

            # Log training loss to WandB
            wandb.log({"epoch": epoch + 1, "train_loss": avg_loss})

            # Save model and evaluate every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_model(epoch + 1)
                test_loss = self.evaluate()
                print(f"Test Loss at Epoch {epoch + 1}: {test_loss}")
                wandb.log({"epoch": epoch + 1, "test_loss": test_loss})

        print("Training complete.")
        wandb.finish()

    def evaluate(self):
        self.model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for ecog_signals, finger_flexions in self.test_loader:
                ecog_signals = ecog_signals.unsqueeze(2)  # Add channel dimension
                ecog_signals, finger_flexions = ecog_signals.to(self.device), finger_flexions.to(self.device)
                outputs = self.model(ecog_signals)
                loss = self.criterion(outputs, finger_flexions.unsqueeze(2))
                test_loss += loss.item()

        return test_loss / len(self.test_loader)

    def save_model(self, epoch):
        checkpoint_path = os.path.join("checkpoints", f'model_epoch_{epoch}.pth')
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved at {checkpoint_path}")

        # Log model checkpoint to WandB
        wandb.save(checkpoint_path)


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

    train_dataset = BCIDataset(_train_file_paths, mode='train', num_channels=48, normalize=True)
    train_stats = (train_dataset.mean, train_dataset.std, train_dataset.label_min, train_dataset.label_max)
    print(f"Training stats: {train_stats}")
    test_dataset = BCIDataset(_test_file_paths, test_label_paths=_test_label_paths, mode='test', num_channels=48, normalize=True, train_stats=train_stats)

    batch_size = 64 * 600
    print(f"Batch size: {batch_size}")
    _train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    _test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    _model = UNet1D(in_channels=48, out_channels=5)
    trainer = Trainer(_model, _train_loader, _test_loader, num_epochs=50, learning_rate=0.001)
    trainer.train()

    final_test_loss = trainer.evaluate()
    print(f"Final Test Loss: {final_test_loss}")
    wandb.log({"final_test_loss": final_test_loss})
