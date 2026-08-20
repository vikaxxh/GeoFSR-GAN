import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class BaselineTrainer:
    """
    Lightweight, CPU-friendly trainer for spatial baseline super-resolution models.
    """
    def __init__(self, model, train_dataset, val_dataset, config):
        self.config = config
        self.device = torch.device("cpu")
        if config["project"]["device"] == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")

        self.model = model.to(self.device)
        self.batch_size = config["training"]["batch_size"]
        self.epochs = config["training"]["epochs"]
        self.lr = config["training"]["lr"]
        self.checkpoint_dir = config["training"]["checkpoint_dir"]
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        self.val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

        self.criterion = nn.L1Loss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=config["training"]["betas"])

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        start_time = time.time()

        for step, batch in enumerate(self.train_loader):
            lr = batch["lr"].to(self.device)
            hr = batch["hr"].to(self.device)

            self.optimizer.zero_grad()
            sr = self.model(lr)
            loss = self.criterion(sr, hr)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(self.train_loader)
        elapsed = time.time() - start_time
        print(f"[Epoch {epoch+1}/{self.epochs}] Train L1 Loss: {avg_loss:.5f} | Time: {elapsed:.2f}s")
        return avg_loss

    def train(self):
        print(f"[Trainer] Starting training for {self.epochs} epochs on device '{self.device}'...")
        best_loss = float("inf")

        for epoch in range(self.epochs):
            loss = self.train_epoch(epoch)
            if loss < best_loss:
                best_loss = loss
                best_model_path = os.path.join(self.checkpoint_dir, "best_model.pth")
                torch.save(self.model.state_dict(), best_model_path)

        last_model_path = os.path.join(self.checkpoint_dir, "last_model.pth")
        torch.save(self.model.state_dict(), last_model_path)
        print(f"[Trainer] Training complete. Best model saved to '{self.checkpoint_dir}/best_model.pth'.")
