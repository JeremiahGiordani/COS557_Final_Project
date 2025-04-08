import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, f1_score
import matplotlib.pyplot as plt
import numpy as np

from dataset import TARXrayDataset
from model import TARRevisionClassifier


image_dir = "/data/home/cos557/jg0037/rothman/images"
csv_path = "/data/home/cos557/jg0037/rothman/parsed_xray_files_log.csv"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


full_dataset = TARXrayDataset(image_dir=image_dir, csv_path=csv_path, transform=transform)

# Split: 70% train, 15% val, 15% test
total_size = len(full_dataset)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

train_set, val_set, test_set = random_split(full_dataset, [train_size, val_size, test_size])

def variable_length_collate(batch):
    images, labels = zip(*batch)  # list of [N_i, 3, H, W] and list of scalars
    return list(images), torch.tensor(labels, dtype=torch.float32)

train_loader = DataLoader(train_set, batch_size=8, shuffle=True, collate_fn=variable_length_collate)
val_loader = DataLoader(val_set, batch_size=8, collate_fn=variable_length_collate)
test_loader = DataLoader(test_set, batch_size=8, collate_fn=variable_length_collate)

print(f"Dataset sizes: Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)}")

model = TARRevisionClassifier(pretrained=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
pos, neg = full_dataset.count_class_distribution()
print(f"Positive samples: {pos}")
print(f"Negative samples: {neg}")


model_path = "tar_revision_model.pt"
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)

preds, targets = model.infer(test_loader, device=device)

acc = accuracy_score(targets, preds)
f1 = f1_score(targets, preds, zero_division=1)
roc_auc = roc_auc_score(targets, preds)

print(f"Test Accuracy: {acc:.4f}")
print(f"Test F1 Score: {f1:.4f}")
print(f"Test ROC AUC: {roc_auc:.4f}")

