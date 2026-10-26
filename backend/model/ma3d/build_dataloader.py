import os
import copy
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from Read_dataset import *


def dataset_labels(dataset):
    """Class index of every sample, read from the sample list without loading images."""
    if isinstance(dataset, Subset):
        labels = dataset_labels(dataset.dataset)
        return [labels[i] for i in dataset.indices]
    if hasattr(dataset, "label2idx"):
        return [dataset.label2idx[s["emotion"]] for s in dataset.samples]
    return [s["label"] for s in dataset.samples]


def class_counts(labels, num_classes):
    counts = [0] * num_classes
    for label in labels:
        counts[label] += 1
    return counts


def get_dataloaders(args):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        normalize,
        transforms.RandomErasing(p=1, scale=(0.05, 0.05))
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    project_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(project_root, "Datasets")

    if args.data_type == "RAF-DB":
        raf_path = os.path.join(data_dir, "RAF-DB_lmk")
        train_dataset = RAFDataset(raf_path, is_train=True, transform=train_transform)
        val_dataset = RAFDataset(raf_path, is_train=False, transform=val_transform)

    elif args.data_type == "VKIST":
        vkist_root = os.path.join(data_dir, "3087data_subject_split")
        stats_path = os.path.join(vkist_root, "micro_sb_emoca_stats.npz")
        train_dataset = VKIST_Dataset(vkist_root, is_train=True, transform=train_transform, stats_path=stats_path)
        val_dataset = VKIST_Dataset(vkist_root, is_train=False, transform=val_transform, stats_path=stats_path)
    elif args.data_type == "FerPlus":
        ferPlus_root = os.path.join(data_dir, "FER_Plus/fer_plus_05")
        train_dataset = FERPlusDataset(ferPlus_root, split="train", transform=train_transform)
        val_dataset = FERPlusDataset(ferPlus_root, split="test", transform=val_transform)
    elif args.data_type == "Caers":
        caers_root = os.path.join(data_dir, "caers/caers")
        train_dataset = CaersDataset(caers_root, split="train", transform=train_transform)
        val_dataset = CaersDataset(caers_root, split="test", transform=val_transform)
    elif args.data_type == "CheoFaMo":
        CheoFamo_root = os.path.join(data_dir, "CheoFamo")
        train_dataset = CheoFaMo(CheoFamo_root, split="train", transform=train_transform)
        val_dataset = CheoFaMo(CheoFamo_root, split="test", transform=val_transform)
    else:
        cheo_root = os.path.join(data_dir, "cheo_dataset")
        train_dataset = DatasetCheo(json_path=os.path.join(cheo_root, "votes_train.json"), root_dir=cheo_root, transform=train_transform)
        val_dataset = DatasetCheo(json_path=os.path.join(cheo_root, "votes_valid.json"), root_dir=cheo_root, transform=val_transform)

    # Hold out a stratified part of the training set (with test-time transforms) and validate on it
    # instead of the test set, e.g. to choose tau for BalancedMarginAwareCELoss.
    val_split = getattr(args, "val_split", 0.0)
    if val_split > 0:
        labels = dataset_labels(train_dataset)
        train_idx, val_idx = train_test_split(
            list(range(len(labels))),
            test_size=val_split,
            stratify=labels,
            random_state=getattr(args, "split_seed", 0)
        )
        holdout_dataset = copy.copy(train_dataset)
        holdout_dataset.transform = val_transform
        val_dataset = Subset(holdout_dataset, val_idx)
        train_dataset = Subset(train_dataset, train_idx)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    return train_loader, val_loader