import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from onion.dataset import get_loaders
from onion.model import OnionNet


def train(
    data_dir: str = "./Dataset",
    epochs: int = 15,
    lr: float = 0.001,
    batch_size: int = 32,
    save_path: str = "model_weights.pth",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, _, classes = get_loaders(data_dir, batch_size)
    print(f"Classes: {classes}")

    model = OnionNet(num_classes=len(classes)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss /= len(train_loader.dataset)
        train_acc = correct / total * 100

        # --- Validation ---
        model.eval()
        val_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader.dataset)
        val_acc = correct / total * 100

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Onion CNN")
    parser.add_argument("--data-dir", default="./Dataset")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--save-path", default="model_weights.pth")
    args = parser.parse_args()

    train(args.data_dir, args.epochs, args.lr, args.batch_size, args.save_path)
