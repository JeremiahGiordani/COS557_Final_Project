import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

class TARRevisionClassifier(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(in_features, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (B, N, C, H, W) → batch of sets of images
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        features = self.backbone(x)  # (B*N, D)
        features = features.view(B, N, -1)  # (B, N, D)
        avg_features = features.mean(dim=1)  # (B, D)
        out = self.classifier(avg_features)  # (B, 1)
        return out.squeeze(1)

    def fit(self, train_loader, val_loader, epochs, optimizer, criterion, device):
        self.to(device)

        for epoch in range(epochs):
            self.train()
            total_loss = 0.0
            all_preds = []
            all_targets = []

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for image_sets, y_batch in pbar:
                batch_tensors = [imgs.to(device) for imgs in image_sets]
                padded_batch = torch.nn.utils.rnn.pad_sequence(batch_tensors, batch_first=True)  # [B, N, 3, 224, 224]
                y_batch = y_batch.to(device)

                optimizer.zero_grad()
                outputs = self(padded_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                preds = (outputs > 0.5).float()
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())

                acc = accuracy_score(all_targets, all_preds)
                f1 = f1_score(all_targets, all_preds, zero_division=1)
                precision = precision_score(all_targets, all_preds, zero_division=1)
                recall = recall_score(all_targets, all_preds, zero_division=1)

                pbar.set_postfix({
                    "Loss": f"{loss.item():.4f}",
                    "Acc": f"{acc:.4f}",
                    "F1": f"{f1:.4f}",
                    "Prec": f"{precision:.4f}",
                    "Rec": f"{recall:.4f}"
                })

            epoch_loss = total_loss / len(train_loader)
            epoch_acc = accuracy_score(all_targets, all_preds)
            epoch_f1 = f1_score(all_targets, all_preds, zero_division=1)
            epoch_precision = precision_score(all_targets, all_preds, zero_division=1)
            epoch_recall = recall_score(all_targets, all_preds, zero_division=1)

            print(f"Epoch {epoch+1}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.4f}, F1={epoch_f1:.4f}, "
                f"Prec={epoch_precision:.4f}, Rec={epoch_recall:.4f}")

            if val_loader:
                self.eval()
                val_preds, val_targets = [], []
                val_loss = 0.0

                with torch.no_grad():
                    for image_sets, y_val in val_loader:
                        batch_tensors = [imgs.to(device) for imgs in image_sets]
                        padded_batch = torch.nn.utils.rnn.pad_sequence(batch_tensors, batch_first=True)
                        y_val = y_val.to(device)

                        outputs = self(padded_batch)
                        val_loss += criterion(outputs, y_val).item()
                        preds = (outputs > 0.5).float()

                        val_preds.extend(preds.cpu().numpy())
                        val_targets.extend(y_val.cpu().numpy())

                val_acc = accuracy_score(val_targets, val_preds)
                val_f1 = f1_score(val_targets, val_preds, zero_division=1)
                val_precision = precision_score(val_targets, val_preds, zero_division=1)
                val_recall = recall_score(val_targets, val_preds, zero_division=1)

                print(f"Validation Loss={val_loss / len(val_loader):.4f}, Acc={val_acc:.4f}, "
                    f"F1={val_f1:.4f}, Prec={val_precision:.4f}, Rec={val_recall:.4f}")

    def infer(self, dataloader, device):
        self.eval()
        self.to(device)

        predictions = []
        labels = []
        with torch.no_grad():
            for x_batch, y_batch in dataloader:
                batch_tensors = [imgs.to(device) for imgs in x_batch]  # List of tensors: [N_i, 3, 224, 224]
                padded_batch = torch.nn.utils.rnn.pad_sequence(batch_tensors, batch_first=True)  # (B, max_N, 3, 224, 224)
                outputs = self(padded_batch)
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float().cpu()
                predictions.extend(preds.numpy())
                labels.extend(y_batch.numpy())
        return predictions, labels
