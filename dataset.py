import os
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from torchvision import transforms
from collections import defaultdict

class TARXrayDataset(Dataset):
    def __init__(self, image_dir, csv_path, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.csv_data = pd.read_csv(csv_path)

        self.patient_revision_map = self._build_revision_map()
        self.data_points = self._collect_image_sets()

    def _build_revision_map(self):
        revision_map = defaultdict(list)
        for _, row in self.csv_data.iterrows():
            pid = int(row['patient_id'])
            revision_status = int(row['revision_status'])
            postop_day = int(row['postop_days'])
            revision_map[pid].append((revision_status, postop_day))
        for pid in revision_map:
            revision_map[pid].sort()
        return revision_map

    def _collect_image_sets(self):
        grouped = defaultdict(list)

        for filepath in glob.glob(os.path.join(self.image_dir, "*.png")):
            fname = os.path.basename(filepath)
            try:
                patient_id, revision_status, prepost_status, days_postop, _ = fname.split("_")
                key = (
                    int(patient_id),
                    int(revision_status),
                    int(prepost_status),
                    int(days_postop)
                )
                grouped[key].append(filepath)
            except ValueError:
                continue

        datapoints = []
        for (pid, rev_status, prepost, days), files in grouped.items():
            label = self._determine_label(pid, rev_status)
            datapoints.append({
                "image_paths": sorted(files),
                "label": label,
                "metadata": {
                    "patient_id": pid,
                    "revision_status": rev_status,
                    "prepost_status": prepost,
                    "days_postop": days
                }
            })
        return datapoints

    def _determine_label(self, pid, current_revision_status):
        revisions = sorted([rev for rev, _ in self.patient_revision_map[pid]])
        return int(current_revision_status < max(revisions))

    def __len__(self):
        return len(self.data_points)

    def __getitem__(self, idx):
        entry = self.data_points[idx]
        image_tensors = []
        for path in entry["image_paths"]:
            img = Image.open(path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            image_tensors.append(img)
        images = torch.stack(image_tensors)  # shape: [N, C, H, W]
        return images, torch.tensor(entry["label"], dtype=torch.float32)

    def get_metadata(self, idx):
        entry = self.data_points[idx]
        return {
            **entry["metadata"],
            "num_images": len(entry["image_paths"]),
            "image_paths": entry["image_paths"]
        }

    def count_class_distribution(self):
        pos = sum(1 for dp in self.data_points if dp["label"] == 1)
        neg = sum(1 for dp in self.data_points if dp["label"] == 0)
        return pos, neg
