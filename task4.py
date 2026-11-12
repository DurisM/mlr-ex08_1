"""Task 4: the existing Task 3 dataset, CNN and training loop, with only requested variants.
Run from this folder: .venv/bin/python task4.py compare
Then inspect task4_results/comparison_table.md before choosing combinations.
"""
from pathlib import Path
import csv
import hashlib
import json
import os
import random
import statistics
import sys
import time
import zipfile

os.environ.setdefault("TORCH_HOME", str(Path("task4_results/torch_cache").resolve()))
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import models, transforms

# CPU gives repeatable runs on this small dataset, including across model types.
device = torch.device("cpu")
torch.set_num_threads(1)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
SEEDS = [0, 1, 2]
CAMERA_SEED = 2026
OUT = Path("task4_results")
OUT.mkdir(exist_ok=True)

# These two classes are copied unchanged from Task 3 (cells 54 and 61).
class LeafDataset(Dataset):
    def __init__(self, folder, transform):
        self.transform = transform
        folder_path = Path(folder)
        self.classes = sorted([d.name for d in folder_path.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.samples = []
        for cls_name in self.classes:
            cls_dir = folder_path / cls_name
            for img_path in sorted(cls_dir.glob("*.jpg")):
                self.samples.append((img_path, self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, class_idx = self.samples[i]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, class_idx

class LeafNetBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 6 * 6, 64)
        self.fc2 = nn.Linear(64, 4)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool(torch.relu(self.bn4(self.conv4(x))))
        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Fixed baseline = Task 3d: LeafNetBN, geometry, Adam 0.001, 15 epochs.
# Every result uses final weights, so checkpoint selection is not another change.
BASE = dict(model="cnn", frozen=False, geometry=True, colour=False,
            cosine=False, epochs=15)
MOBILE = dict(BASE, model="mobilenet", geometry=False)
EXPERIMENTS = {
    "baseline": dict(BASE),
    "cnn_plain": dict(BASE, geometry=False),
    "cnn_colour": dict(BASE, colour=True),
    "mobile_frozen": dict(MOBILE, frozen=True),
    "mobile_finetuned": dict(MOBILE),
    "mobile_geometry": dict(MOBILE, geometry=True),
    "mobile_geometry_colour": dict(MOBILE, geometry=True, colour=True),
    "mobile_cosine": dict(MOBILE, cosine=True),
    "mobile_30epochs": dict(MOBILE, epochs=30),
}
# Direct references isolate each change. Also show every row vs the fixed baseline.
REFERENCES = {"cnn_colour": "baseline", "mobile_frozen": "cnn_plain", "cnn_plain": "baseline",
              "mobile_finetuned": "mobile_frozen", "mobile_geometry": "mobile_finetuned",
              "mobile_geometry_colour": "mobile_geometry", "mobile_cosine": "mobile_finetuned",
              "mobile_30epochs": "mobile_finetuned"}
# cnn_plain controls geometry before the transfer-learning comparison.
# Both frozen and fine-tuned MobileNet use identical images and no augmentation.


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def make_model(config, pretrained=True):
    if config["model"] == "cnn":
        return LeafNetBN()
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 4)
    if config["frozen"]:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.classifier[3].parameters():
            parameter.requires_grad_(True)
    return model


def accuracy(model, loader):
    model.eval()
    correct = 0
    predictions = []
    with torch.no_grad():
        for images, labels in loader:
            predicted = model(images.to(device)).argmax(1).cpu()
            correct += (predicted == labels).sum().item()
            predictions.extend(predicted.tolist())
    return correct / len(loader.dataset), predictions


def prepare_data():
    plain = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    train_ds = LeafDataset("data/leaves/train", plain)
    val_ds = LeafDataset("data/leaves/val", plain)
    if train_ds.classes != val_ds.classes or len(train_ds.classes) != 4:
        raise ValueError("Expected the same four species in both splits.")
    if not len(train_ds) or not len(val_ds):
        raise ValueError("The group ZIP must be extracted into data/leaves first.")
    # Check the source files once; never silently resize/replace the group split.
    for path, _ in train_ds.samples + val_ds.samples:
        with Image.open(path) as image:
            if image.size != (96, 96):
                raise ValueError(f"Expected 96 x 96 image: {path}")
    manifest = {"classes": train_ds.classes, "camera_seed": CAMERA_SEED,
                "torch": torch.__version__, "device": str(device), "cpu_threads": torch.get_num_threads(), "seeds": SEEDS,
                "files": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p, _ in train_ds.samples + val_ds.samples}}
    manifest_path = OUT / "data_manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old["files"] != manifest["files"]:
            raise ValueError("Dataset changed: use a separate results folder for a new comparison.")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    camera_path = OUT / "camera_validation.pt"
    if not camera_path.exists():
        set_seed(CAMERA_SEED)
        camera_transform = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=(0.6, 1.0)),
            transforms.RandomRotation(20),
            transforms.ColorJitter(0.25, 0.25, 0.25, 0.05),
            transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
        camera_source = LeafDataset("data/leaves/val", camera_transform)
        images, labels = zip(*[camera_source[i] for i in range(len(camera_source))])
        torch.save({"images": torch.stack(images), "labels": torch.tensor(labels)}, camera_path)
    camera = torch.load(camera_path, map_location="cpu", weights_only=True)
    camera_ds = TensorDataset(camera["images"], camera["labels"])
    assert camera["labels"].tolist() == [label for _, label in val_ds.samples]
    print(f"Data: {len(train_ds)} train, {len(val_ds)} validation; classes {train_ds.classes}", flush=True)
    print(f"One validation image = {100 / len(val_ds):.3f} percentage points.", flush=True)
    return train_ds, val_ds, camera_ds


def train_variant(name, config, seed, val_ds, camera_ds):
    result_path = OUT / f"{name}_seed{seed}.json"
    if result_path.exists() and (OUT / f"{name}_seed{seed}.pt").exists():
        previous = json.loads(result_path.read_text())
        if previous["config"] != config:
            raise ValueError(f"Configuration changed for saved run {name}; use a new name.")
        print(f"Already measured: {name} seed {seed}", flush=True)
        return previous["result"]
    set_seed(seed)
    operations = []
    if config["geometry"]:
        operations = [transforms.RandomResizedCrop(96, scale=(0.6, 1.0)),
                      transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
                      transforms.RandomRotation(20)]
    if config["colour"]:
        operations.append(transforms.ColorJitter(0.25, 0.25, 0.25, 0.05))
    operations += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    train_ds = LeafDataset("data/leaves/train", transforms.Compose(operations))
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0,
                              generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    camera_loader = DataLoader(camera_ds, batch_size=32, shuffle=False, num_workers=0)
    model = make_model(config).to(device)
    # Same random stream for augmentation regardless of model initialization.
    set_seed(seed)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
                 if config["cosine"] else None)
    history = []
    start = time.perf_counter()
    for epoch in range(config["epochs"]):
        model.train()
        if config["frozen"]:
            # Freeze pretrained BN running statistics and dropout as well as weights.
            model.eval()
            model.classifier[3].train()
        loss_sum = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(images)
        val_accuracy, _ = accuracy(model, val_loader)
        history.append(dict(epoch=epoch + 1, loss=loss_sum / len(train_ds),
                            clean_accuracy=val_accuracy, lr=optimizer.param_groups[0]["lr"]))
        if scheduler is not None:
            scheduler.step()
        print(f"{name} seed {seed} epoch {epoch+1:2d}: loss {history[-1]['loss']:.4f}, "
              f"clean {val_accuracy:.3f}", flush=True)
    clean, clean_predictions = accuracy(model, val_loader)
    camera, camera_predictions = accuracy(model, camera_loader)
    row = dict(variant=name, seed=seed, clean_accuracy=clean, camera_accuracy=camera,
               epochs=config["epochs"], parameters=sum(p.numel() for p in model.parameters()),
               seconds=time.perf_counter() - start)
    torch.save({"state_dict": model.cpu().state_dict(), "config": config, "seed": seed,
                "classes": val_ds.classes}, OUT / f"{name}_seed{seed}.pt")
    result_path.write_text(json.dumps(dict(result=row, config=config, history=history,
        clean_predictions=clean_predictions, camera_predictions=camera_predictions,
        labels=[label for _, label in val_ds.samples]), indent=2))
    print("RESULT", row, flush=True)
    return row


def summarize():
    rows = [json.loads(p.read_text())["result"] for p in sorted(OUT.glob("*_seed*.json"))]
    with open(OUT / "results.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    def mean_sd(values):
        return f"{statistics.mean(values):.2f} ± {statistics.stdev(values):.2f}"
    by_run = {(r["variant"], r["seed"]): r for r in rows}
    lines = ["| Variant (3 seeds) | Clean % ± SD | Camera % ± SD | Δ clean vs baseline, pp | Δ camera vs baseline, pp | Direct reference | Paired Δ clean, pp ± SD | Paired Δ camera, pp ± SD |",
             "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |"]
    for name in dict.fromkeys(row["variant"] for row in rows):
        group = [r for r in rows if r["variant"] == name]
        if len(group) != len(SEEDS):
            continue
        direct = REFERENCES.get(name, "baseline")
        values = [mean_sd([100*r[key] for r in group]) for key in ["clean_accuracy", "camera_accuracy"]]
        for key in ["clean_accuracy", "camera_accuracy"]:
            values.append(f"{statistics.mean(100*(r[key]-by_run['baseline',r['seed']][key]) for r in group):+.2f}")
        values.append(direct)
        for key in ["clean_accuracy", "camera_accuracy"]:
            values.append(mean_sd([100*(r[key]-by_run[direct,r['seed']][key]) for r in group]))
        lines.append(f"| {name} | " + " | ".join(values) + " |")
    table = "\n".join(lines) + "\n"
    (OUT / "comparison_table.md").write_text(table)
    print(table, flush=True)
    return rows


def export_phone(val_ds, camera_ds):
    import onnx
    import onnxruntime as ort
    selection = json.loads((OUT / "selection.json").read_text())
    name, seed = selection["variant"], selection["seed"]
    checkpoint = torch.load(OUT / f"{name}_seed{seed}.pt", map_location="cpu", weights_only=True)
    model = make_model(checkpoint["config"], pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval().cpu()
    folder = OUT / "phone"
    folder.mkdir(exist_ok=True)
    onnx_path = folder / "leaf_model.onnx"
    # Static shape and opset 17 also support the course app's browser runtime.
    torch.onnx.export(model, (torch.zeros(1, 3, 96, 96),), str(onnx_path),
                      input_names=["image"], output_names=["logits"],
                      opset_version=17, dynamo=False, external_data=False)
    info = dict(classes=val_ds.classes, input_size=96, mean=MEAN, std=STD)
    exported = onnx.load(str(onnx_path))
    exported.metadata_props.add(key="model_info", value=json.dumps(info))
    onnx.checker.check_model(exported)
    onnx.save(exported, str(onnx_path))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(onnx_path), sess_options=options, providers=["CPUExecutionProvider"])
    maximum_error = 0.0
    # Static batch size 1 matches the phone; check every clean and camera-like image.
    for dataset in [val_ds, camera_ds]:
        for image, _ in dataset:
            x = image.unsqueeze(0)
            with torch.no_grad():
                expected = model(x).softmax(1).numpy()
            logits = session.run(["logits"], {"image": x.numpy()})[0]
            actual = np.exp(logits - logits.max(axis=1, keepdims=True))
            actual /= actual.sum(axis=1, keepdims=True)
            maximum_error = max(maximum_error, float(np.max(np.abs(actual - expected))))
            np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4)
    Image.open(val_ds.samples[0][0]).convert("RGB").save(folder / "check_image.jpg", quality=95)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    # Lossless copy lets the course app check the exact same decoded reference pixels.
    Image.open(folder / "check_image.jpg").convert("RGB").save(folder / "check_image.png")
    x = transform(Image.open(folder / "check_image.jpg").convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        expected = model(x).softmax(1)[0].numpy()
    logits = session.run(["logits"], {"image": x.numpy()})[0][0]
    actual = np.exp(logits - logits.max()); actual /= actual.sum()
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4)
    (folder / "model_info.json").write_text(json.dumps(info, indent=2))
    (folder / "check_expected.json").write_text(json.dumps(dict(zip(val_ds.classes, expected.tolist())), indent=2))
    validation = dict(variant=name, seed=seed, validation_images=2*len(val_ds),
                      max_probability_error=maximum_error, bytes=onnx_path.stat().st_size,
                      reference_image_error=float(np.max(np.abs(actual - expected))))
    (folder / "export_validation.json").write_text(json.dumps(validation, indent=2))
    with zipfile.ZipFile(OUT / "ex08_task4_for_agent.zip", "w") as archive:
        for filename in ["leaf_model.onnx", "model_info.json", "check_image.jpg", "check_expected.json"]:
            archive.write(folder / filename, arcname=filename)
    print("EXPORT VERIFIED", validation, flush=True)


if __name__ == "__main__":
    _, val_ds, camera_ds = prepare_data()
    if (OUT / "combination_plan.json").exists():
        for name, entry in json.loads((OUT / "combination_plan.json").read_text()).items():
            REFERENCES[name] = entry["reference"]
    command = sys.argv[1] if len(sys.argv) > 1 else "compare"
    if command == "compare":
        for name, config in EXPERIMENTS.items():
            for seed in SEEDS:
                train_variant(name, config, seed, val_ds, camera_ds)
        summarize()
    elif command == "combine":
        plan = json.loads((OUT / "combination_plan.json").read_text())
        for name, entry in plan.items():
            REFERENCES[name] = entry["reference"]
            for seed in SEEDS:
                train_variant(name, entry["config"], seed, val_ds, camera_ds)
        summarize()
    elif command == "export":
        export_phone(val_ds, camera_ds)
    elif command == "summary":
        summarize()
    else:
        raise ValueError("Use compare, combine, summary or export.")
