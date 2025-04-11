import os
import glob
import torch
from torch.utils.data import Dataset, Subset
from PIL import Image
import pandas as pd
from torchvision import transforms
from collections import defaultdict
import random


class TARXrayDataset(Dataset):
    def __init__(self, image_dir, csv_path, patient_info_path=None, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.csv_data = pd.read_csv(csv_path)
        self.patient_info_path=patient_info_path
        self.patient_info = pd.read_csv(patient_info_path) if patient_info_path else None

        self.metadata_keys = [
            "Age",
            "BMI",
            "sex (m=0, f=1)",
            "days_postop",
            "Time To Surgery/Repair (days after injury, approximately) (according to first encounter with surgeon note i.e. \"patient reports ongoing pain for 2 years\" -> (day of appt- 2 years) to the day of SXO) "
        ]
        self.metadata_means = {}
        self.metadata_stds = {}

        self.patient_revision_map = self._build_revision_map()
        self.data_points = self._collect_image_sets()
        self._compute_normalization_stats()

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
                if len(grouped[key]) > 8:
                    continue
                grouped[key].append(filepath)
            except ValueError:
                continue

        datapoints = []
        for (pid, rev_status, prepost, days), files in grouped.items():
            label = self._determine_label(pid, rev_status)
            metadata = {
                "patient_id": pid,
                "revision_status": rev_status,
                "prepost_status": prepost,
                "days_postop": days
            }

            if self.patient_info_path is not None:
                extra_info = self.patient_info[self.patient_info['ID'] == pid]
                if not extra_info.empty:
                    metadata.update(extra_info.iloc[0].to_dict())

            datapoints.append({
                "image_paths": sorted(files),
                "label": label,
                "metadata": metadata
            })
        return datapoints

    def _compute_normalization_stats(self):
        collected = {k: [] for k in self.metadata_keys}
        for dp in self.data_points:
            for k in self.metadata_keys:
                val = dp["metadata"].get(k, 0.0)
                try:
                    val = float(val)
                    if pd.isna(val):
                        val = 0.0
                except (ValueError, TypeError):
                    val = 0.0
                collected[k].append(val)

        for k in self.metadata_keys:
            values = collected[k]
            mean = sum(values) / len(values)
            std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
            self.metadata_means[k] = mean
            self.metadata_stds[k] = std if std > 0 else 1.0

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
        images = torch.stack(image_tensors)
        if self.patient_info_path:
            metadata_tensor = self._extract_numerical_metadata(entry["metadata"])
            return images, torch.tensor(entry["label"], dtype=torch.float32), metadata_tensor
        else:
            return images, torch.tensor(entry["label"], dtype=torch.float32)

    def _extract_numerical_metadata(self, metadata):
        values = []
        for k in self.metadata_keys:
            val = metadata.get(k, 0.0)
            try:
                val = float(val)
                if pd.isna(val):
                    val = 0.0
            except (ValueError, TypeError):
                val = 0.0
            # Normalize
            mean = self.metadata_means[k]
            std = self.metadata_stds[k]
            val = (val - mean) / std
            values.append(val)
        return torch.tensor(values, dtype=torch.float32)

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

    def split_by_patient(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
        pid_to_indices = defaultdict(list)
        for idx, dp in enumerate(self.data_points):
            pid = dp["metadata"]["patient_id"]
            pid_to_indices[pid].append(idx)

        patient_ids = list(pid_to_indices.keys())
        rng = random.Random(seed)
        rng.shuffle(patient_ids)

        total_patients = len(patient_ids)
        train_end = int(train_ratio * total_patients)
        val_end = train_end + int(val_ratio * total_patients)

        train_pids = set(patient_ids[:train_end])
        val_pids = set(patient_ids[train_end:val_end])
        test_pids = set(patient_ids[val_end:])

        def collect_indices(pids):
            return [idx for pid in pids for idx in pid_to_indices[pid]]

        train_indices = collect_indices(train_pids)
        val_indices = collect_indices(val_pids)
        test_indices = collect_indices(test_pids)

        def print_stats(name, indices):
            labels = [int(self.data_points[i]["label"]) for i in indices]
            pos = sum(labels)
            neg = len(labels) - pos
            print(f"{name} Set: Total={len(labels)}, Positives={pos}, Negatives={neg}")

        print_stats("Train", train_indices)
        print_stats("Val", val_indices)
        print_stats("Test", test_indices)

        return (
            Subset(self, train_indices),
            Subset(self, val_indices),
            Subset(self, test_indices),
        )
