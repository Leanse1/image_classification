import json
import math
import torch
import timm
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim import AdamW
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt


# =========================
# HYPERPARAMETERS
# =========================
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-4
WEIGHT_DECAY = 1e-4
MODEL_NAME = "efficientnet_lite0"   
WARMUP_EPOCHS = 7
FINE_TUNE_RATIO = 0.4

USE_MIXUP = True
MIXUP_ALPHA = 0.3

USE_CLASS_WEIGHTS = True


# =========================
# DATASET
# =========================
data_dir = "/home/leanse/AI/interview/clearquote/dataset/"
train_dir = data_dir + "train"
val_dir   = data_dir + "val"
test_dir  = data_dir + "test"

train_tf = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomRotation(8),
    transforms.RandomPerspective(distortion_scale=0.1, p=0.4),
    transforms.ColorJitter(0.25,0.25,0.15,0.03),
    transforms.GaussianBlur(3),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

val_tf = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
val_ds   = datasets.ImageFolder(val_dir,   transform=val_tf)
test_ds  = datasets.ImageFolder(test_dir,  transform=val_tf)

num_classes = len(train_ds.classes)
print("Classes:", train_ds.classes)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using:", DEVICE)


# =========================
# MODEL
# =========================
model = timm.create_model(MODEL_NAME, pretrained=True)
in_features = model.classifier.in_features
model.classifier = nn.Linear(in_features, num_classes)
model = model.to(DEVICE)


# =========================
# CLASS WEIGHTS
# =========================
if USE_CLASS_WEIGHTS:
    labels = [y for _, y in train_ds.samples]
    class_counts = Counter(labels)
    total = sum(class_counts.values())
    weights = [total/class_counts[i] for i in range(num_classes)]
    class_weights = torch.tensor(weights, dtype=torch.float).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
else:
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)


optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=1e-3,
    steps_per_epoch=len(train_loader),
    epochs=EPOCHS
)


# =========================
# FREEZE & UNFREEZE
# =========================
def freeze_backbone():
    for name, param in model.named_parameters():
        param.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True

def unfreeze_last_layers(ratio=0.4):
    params = list(model.parameters())
    k = int(len(params) * (1 - ratio))
    for p in params[k:]:
        p.requires_grad = True

freeze_backbone()


# =========================
# MIXUP
# =========================
def mixup(x, y, alpha=MIXUP_ALPHA):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0)).to(DEVICE)
    mixed_x = lam*x + (1-lam)*x[idx, :]
    return mixed_x, y, y[idx], lam


# =========================
# METRICS
# =========================
def precision_recall(logits, target):
    preds = logits.argmax(1)

    tp = (preds == target).sum().item()
    fp = (preds != target).sum().item()
    fn = fp

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)

    return precision, recall


# =========================
# EVALUATION
# =========================
def evaluate(dataloader):
    model.eval()
    loss_sum, correct, total = 0, 0, 0
    p_sum, r_sum, count = 0, 0, 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)

            loss_sum += criterion(out, y).item()
            pred = out.argmax(1)
            correct += pred.eq(y).sum().item()
            total += y.size(0)

            p, r = precision_recall(out, y)
            p_sum += p
            r_sum += r
            count += 1

    return (
        loss_sum / len(dataloader),
        (correct / total) * 100,
        p_sum / count,
        r_sum / count
    )


# =========================
# TRAIN
# =========================
best_acc = 0
best_val_loss = float("inf")
epochs_no_improve = 0
EARLY_STOPPING_PATIENCE = 7

train_losses = []
val_losses   = []
train_accs   = []
val_accs     = []

for epoch in range(EPOCHS):

    if epoch == WARMUP_EPOCHS:
        print("Unfreezing last layers...")
        unfreeze_last_layers(FINE_TUNE_RATIO)

    model.train()
    correct, total = 0, 0
    running_loss = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()

        if USE_MIXUP:
            imgs, ya, yb, lam = mixup(imgs, labels)
            outputs = model(imgs)
            loss = lam * criterion(outputs, ya) + (1-lam)*criterion(outputs, yb)
        else:
            outputs = model(imgs)
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    scheduler.step()

    train_losses.append(running_loss / len(train_loader))
    train_accs.append((correct/total)*100)

    val_loss, val_acc, prec, rec = evaluate(val_loader)
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    print(f"""
        Epoch {epoch+1}/{EPOCHS}
        Train Loss: {train_losses[-1]:.4f} | Train Acc: {train_accs[-1]:.2f}%
        Val Loss  : {val_loss:.4f} | Val Acc  : {val_acc:.2f}%
        Precision : {prec:.3f} | Recall: {rec:.3f}
        """)

    # =========================
    # Early Stopping & Save Best Model
    # =========================
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), "efficientnet_lite_best.pth")
        print("Saved Best Model")
    else:
        epochs_no_improve += 1
        print(f"No improvement for {epochs_no_improve} epochs")

    if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
        print(f"Early stopping triggered after {epoch+1} epochs")
        break

print("Training Done 🎯")



# =========================
# TEST EVALUATION
# =========================
print("\nEvaluating on Test Set...")
test_loss, test_acc, p, r = evaluate(test_loader)

print(f"""
Test Loss : {test_loss:.4f}
Test Acc  : {test_acc:.2f}%
Precision : {p:.3f}
Recall    : {r:.3f}
""")


import onnxruntime as ort
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
onnx_model = "efficientnet_best.onnx"

# Create session
session = ort.InferenceSession(onnx_model, providers=["CPUExecutionProvider"])

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

all_preds = []
all_labels = []


def softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


model.eval()

with torch.no_grad():
    for imgs, labels in test_loader:      # <-- you already defined test_loader
        imgs = imgs.numpy()               # to numpy
        outputs = session.run(
            [output_name],
            {input_name: imgs}
        )[0]

        probs = softmax(outputs)
        preds = np.argmax(probs, axis=1)

        all_preds.extend(preds)
        all_labels.extend(labels.numpy())


all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

acc = (all_preds == all_labels).mean() * 100
print(f"\nTest Accuracy: {acc:.2f}%")

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=test_ds.classes))

print("\nConfusion Matrix:")
print(confusion_matrix(all_labels, all_preds))
