import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from vit_model import vit_tiny
from train import train_epoch, evaluate
from data import get_cifar10_loaders

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vision Transformer (ViT) CIFAR-10 Classifier")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--model-path", type=str, default="vit_cifar10.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    train_loader, test_loader = get_cifar10_loaders(args.data_dir, args.batch_size)
    model = vit_tiny(num_classes=10).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        logger.info("Epoch %d | train_loss=%.4f val_loss=%.4f val_acc=%.4f",
                    epoch, train_loss, val_loss, val_acc)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), args.model_path)
            logger.info("Saved best model (acc=%.4f) to %s", best_acc, args.model_path)


if __name__ == "__main__":
    main()
