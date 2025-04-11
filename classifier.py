import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
import gc
import os
import psutil

def get_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def get_virtual_mem_mb():
    return psutil.Process(os.getpid()).memory_info().vms / 1024 / 1024

class AttentionalPooling(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, features):  # features: [B, N, D]
        scores = self.attn(features).squeeze(-1)         # [B, N]
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # [B, N, 1]
        pooled = torch.sum(features * weights, dim=1)    # [B, D]
        return pooled


class TARRevisionClassifier(nn.Module):
    def __init__(self, pretrained=True, use_attention_pooling=False, metadata_dim=None):
        super().__init__()
        self.use_attention_pooling = use_attention_pooling
        self.metadata_dim = metadata_dim

        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        if use_attention_pooling:
            self.pooling = AttentionalPooling(in_features)
        else:
            self.pooling = lambda x: x.mean(dim=1)

        classifier_input_dim = in_features + (metadata_dim if metadata_dim else 0)
        self.classifier = nn.Linear(classifier_input_dim, 1)

    def forward(self, x, metadata=None):
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        features = self.backbone(x)
        features = features.view(B, N, -1)
        pooled = self.pooling(features)

        if metadata is not None:
            pooled = torch.cat([pooled, metadata], dim=1)

        out = self.classifier(pooled)
        return out.squeeze(1)

    def fit(self, train_loader, val_loader, epochs, optimizer, criterion, device, model_path):
        self.to(device)
        best_val_precision = 0.0

        for epoch in range(epochs):
            self.train()
            total_loss = 0.0
            running_acc, running_f1, running_precision, running_recall = [], [], [], []

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for image_sets, y_batch, metadata_batch in pbar:
                batch_tensors = [imgs.to(device) for imgs in image_sets]
                padded_batch = torch.nn.utils.rnn.pad_sequence(batch_tensors, batch_first=True)
                y_batch = y_batch.to(device)
                metadata_batch = metadata_batch.to(device) if metadata_batch is not None else None

                optimizer.zero_grad()
                outputs = self(padded_batch, metadata_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                gc.collect()
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()

                acc = accuracy_score(y_batch.cpu(), preds.cpu())
                f1 = f1_score(y_batch.cpu(), preds.cpu(), zero_division=1)
                precision = precision_score(y_batch.cpu(), preds.cpu(), zero_division=1)
                recall = recall_score(y_batch.cpu(), preds.cpu(), zero_division=1)

                running_acc.append(acc)
                running_f1.append(f1)
                running_precision.append(precision)
                running_recall.append(recall)

                pbar.set_postfix({
                    "Loss": f"{loss.item():.4f}",
                    "Acc": f"{np.mean(running_acc):.4f}",
                    "F1": f"{np.mean(running_f1):.4f}",
                    "Prec": f"{np.mean(running_precision):.4f}",
                    "Rec": f"{np.mean(running_recall):.4f}",
                    "MemMB": f"{get_memory_mb():.1f}",
                    "VIRT": f"{get_virtual_mem_mb():.1f} MB"
                })
                del outputs, loss, probs, preds, batch_tensors, padded_batch, y_batch, metadata_batch
                gc.collect()
            print("OUTSIDE OF FOR LOOP")
            epoch_loss = total_loss / len(train_loader)
            epoch_acc = np.mean(running_acc)
            epoch_f1 = np.mean(running_f1)
            epoch_precision = np.mean(running_precision)
            epoch_recall = np.mean(running_recall)

            print(f"Epoch {epoch+1}: Loss={epoch_loss:.4f}, Accuracy={epoch_acc:.4f}, F1={epoch_f1:.4f}, Precision={epoch_precision:.4f}, Recall={epoch_recall:.4f}")

            if val_loader:
                self.eval()
                val_loss = 0.0
                running_acc, running_f1, running_precision, running_recall = [], [], [], []

                with torch.no_grad():
                    for image_sets, y_val, metadata_batch in val_loader:
                        batch_tensors = [imgs.to(device) for imgs in image_sets]
                        padded_batch = torch.nn.utils.rnn.pad_sequence(batch_tensors, batch_first=True)
                        y_val = y_val.to(device)
                        metadata_batch = metadata_batch.to(device) if metadata_batch is not None else None

                        outputs = self(padded_batch, metadata_batch)
                        val_loss += criterion(outputs, y_val).item()

                        probs = torch.sigmoid(outputs)
                        preds = (probs > 0.5).float()

                        acc = accuracy_score(y_val.cpu(), preds.cpu())
                        f1 = f1_score(y_val.cpu(), preds.cpu(), zero_division=1)
                        precision = precision_score(y_val.cpu(), preds.cpu(), zero_division=1)
                        recall = recall_score(y_val.cpu(), preds.cpu(), zero_division=1)

                        running_acc.append(acc)
                        running_f1.append(f1)
                        running_precision.append(precision)
                        running_recall.append(recall)

                        del outputs, probs, preds, batch_tensors, padded_batch, metadata_batch, y_val
                        gc.collect()

                val_loss /= len(val_loader)
                val_acc = np.mean(running_acc)
                val_f1 = np.mean(running_f1)
                val_precision = np.mean(running_precision)
                val_recall = np.mean(running_recall)

                print(f"Validation Loss={val_loss:.4f}, Acc={val_acc:.4f}, "
                    f"F1={val_f1:.4f}, Prec={val_precision:.4f}, Rec={val_recall:.4f}")

                if val_precision >= best_val_precision:
                    torch.save(self.state_dict(), model_path)
                    print(f"Model saved to {model_path}")

    def infer(self, dataloader, device):
        self.eval()
        self.to(device)

        running_acc, running_f1, running_precision, running_recall = [], [], [], []

        with torch.no_grad():
            for x_batch, y_batch, metadata_batch in dataloader:
                batch_tensors = [imgs.to(device) for imgs in x_batch]
                padded_batch = torch.nn.utils.rnn.pad_sequence(batch_tensors, batch_first=True)
                metadata_batch = metadata_batch.to(device) if metadata_batch is not None else None
                y_batch = y_batch.to(device)

                outputs = self(padded_batch, metadata_batch)
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()

                acc = accuracy_score(y_batch.cpu(), preds.cpu())
                f1 = f1_score(y_batch.cpu(), preds.cpu(), zero_division=1)
                precision = precision_score(y_batch.cpu(), preds.cpu(), zero_division=1)
                recall = recall_score(y_batch.cpu(), preds.cpu(), zero_division=1)

                running_acc.append(acc)
                running_f1.append(f1)
                running_precision.append(precision)
                running_recall.append(recall)
                del outputs, probs, preds, batch_tensors, padded_batch, metadata_batch, y_batch
                gc.collect()

        print(f"Inference Metrics:")
        print(f"Accuracy:  {np.mean(running_acc):.4f}")
        print(f"F1 Score:  {np.mean(running_f1):.4f}")
        print(f"Precision: {np.mean(running_precision):.4f}")
        print(f"Recall:    {np.mean(running_recall):.4f}")

