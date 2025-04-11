import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, f1_score
import matplotlib.pyplot as plt
import numpy as np

from dataset import TARXrayDataset
from classifier import TARRevisionClassifier

def variable_length_collate(batch):
    if len(batch[0]) == 3:  # if metadata is included
        images, labels, metadata = zip(*batch)
        metadata_tensor = torch.stack(metadata)  # shape: [B, metadata_dim]
        return list(images), torch.tensor(labels, dtype=torch.float32), metadata_tensor
    else:
        images, labels = zip(*batch)
        return list(images), torch.tensor(labels, dtype=torch.float32)


if __name__ == "__main__":
    image_dir = "/data/home/cos557/jg0037/rothman/images"
    csv_path = "/data/home/cos557/jg0037/rothman/parsed_xray_files_log.csv"
    patient_info_path = "/data/home/cos557/jg0037/rothman/TAR_Sheet_fo_stats_SGP_7_9_24_output4.csv"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    full_dataset = TARXrayDataset(image_dir=image_dir, csv_path=csv_path, transform=transform, patient_info_path=patient_info_path)

    _, _, sample_metadata = full_dataset[0]
    metadata_dim = sample_metadata.shape[0]
    print(f"Metadata feature dimension: {metadata_dim}")

    # Split: 70% train, 15% val, 15% test
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    # seed = 42
    # generator = torch.Generator().manual_seed(seed)

    # train_set, val_set, test_set = random_split(full_dataset, [train_size, val_size, test_size], generator=generator)

    # train_loader = DataLoader(train_set, batch_size=8, shuffle=True, collate_fn=variable_length_collate, num_workers=0, pin_memory=False)
    # val_loader = DataLoader(val_set, batch_size=8, collate_fn=variable_length_collate, num_workers=0, pin_memory=False)
    # test_loader = DataLoader(test_set, batch_size=8, collate_fn=variable_length_collate, num_workers=0, pin_memory=False)

    seed=56
    train_set, val_set, test_set = full_dataset.split_by_patient(seed=seed)
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, collate_fn=variable_length_collate, num_workers=1, pin_memory=False)
    val_loader = DataLoader(val_set, batch_size=8, collate_fn=variable_length_collate, num_workers=1, pin_memory=False)
    test_loader = DataLoader(test_set, batch_size=8, collate_fn=variable_length_collate, num_workers=1, pin_memory=False)

    print(f"Dataset sizes: Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)}")

    model = TARRevisionClassifier(
        pretrained=True,
        use_attention_pooling=False,
        metadata_dim=metadata_dim
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    pos, neg = full_dataset.count_class_distribution()
    print(f"Positive samples: {pos}")
    print(f"Negative samples: {neg}")

    model_path = "model_with_metadata.pt"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    model.infer(test_loader, device=device)

